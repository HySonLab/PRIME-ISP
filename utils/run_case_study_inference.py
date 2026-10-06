"""
run_case_study_inference.py

Runs PRIME-ISP inference on the case-study targets produced by
extract_case_study_labels.py (thrombin/8TQS, er_alpha/3L03, calmodulin/1CDL),
one prediction per (target, site) pair -- e.g. thrombin has two sites
(DPI_exosite_II, SMPI_active_site), each scored with the checkpoint for its
own task, against that site's own label set. All sites of the same target
share ONE input structure, so the model's ability to tell them apart from
structure alone is exactly what's being tested.

WHY THIS DOESN'T REUSE render_prime_predictions.py's labels_from_positions:
that helper matches ground-truth positions to residues by residue NUMBER
ONLY (via "chain:resnum".split(":")[1]), which is fine for the single-chain
tasks it was written for, but WRONG here: thrombin's light (~1-36) and
heavy (~1-259) chains, and HIV-1 RT's p66 (~600-1153) and p51 (~600-1027)
chains, all share overlapping auth_seq_id ranges (RT's overlap almost
entirely). This script's label matching therefore uses the full
"chain:resnum" string, matching extract_case_study_labels.py's
positions_by_chain output exactly.

Pipeline per target:
  1. Build ONE hierarchical graph from {target}_clean.pdb (surface export +
     ESM-2 + all graph levels) -- built once per target and reused across
     that target's sites, since it's the same input structure.
  2. Per site: load (and cache) the checkpoint ensemble for that site's
     task, build a chain-aware residue_labels tensor from the site's
     positions_by_chain, run inference, compute TP/TN/FP/FN metrics at a
     single threshold.
  3. Save the graph (with that site's residue_labels baked in) to
     --graph_dir/{target}__{site}.pt and copy the clean PDB to
     --task_dir/PDBs/{target}__{site}.pdb, so your EXISTING
     render_prime_predictions.py / visualize_binding_site.py can render it
     completely unchanged via:
         --record_id {target}__{site} --graph_dir <graph_dir> --task_dir <task_dir>
  4. Save metrics + residue-level predictions as JSON under --output_dir.

PLEASE VERIFY the CASE_STUDY_SITE_TASKS mapping below against your actual
data_config.yaml task keys / checkpoint filenames before running -- I've
mapped each site to the interaction-class task that matches it, but the
exact task-name variant that maps to a checkpoint you've actually trained
(e.g. PepPI has two: 'PepPI-site_Train640-Test639' vs.
'PepPI-site_Train1154-Test125') is a call you should make, not one that
should be guessed silently.

Usage:
    python utils/run_case_study_inference.py \
        --pdb_dir ./Datasets/CaseStudies/clean_pdbs \
        --labels_dir ./Datasets/CaseStudies/labels \
        --data_config ./config/data_config.yaml \
        --model_config ./config/model_config.yaml \
        --graph_dir ./Datasets/CaseStudies/Graphs \
        --task_dir ./Datasets/CaseStudies \
        --output_dir ./Datasets/CaseStudies/predictions \
        --k_folds 5
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import PRIME
from utils.helpers import load_config
from utils.export_surface import export_surface
from utils.hierarchical_graph import load_protein_from_pdb, build_hierarchical_protein_graph

# ======================================================
# Which task (=> which checkpoint) evaluates which site.
# VERIFY against your data_config.yaml keys / trained checkpoints.
# ======================================================
CASE_STUDY_SITE_TASKS = {
    "thrombin": {
        "DPI_exosite_II":   "DPI-site",
        "SMPI_active_site": "SMPI-site",
    },
    "er_alpha": {
        "SMPI_orthosteric":   "SMPI-site",
        "PepPI_coactivator":  "PepPI-site_Train1154-Test125",
    },
    "calmodulin": {
        "MPI_calcium": "MPI-site_CA",
        "PepPI_target": "PepPI-site_Train1154-Test125",
    },
}


# ======================================================
# Model build / load (mirrors render_prime_predictions.py's pattern)
# ======================================================

def build_model(model_config, task, num_classes, active_levels, readout_level, device):
    return PRIME(
        num_classes=num_classes,
        input_dims=model_config["hierarchical"]["input_dims"],
        active_levels=active_levels,
        readout_level=readout_level,
        hidden_dim=model_config["hierarchical"]["hidden_dim"],
        encoder_layers=model_config["hierarchical"]["n_layers"],
        head_hidden_dim=model_config["head"][task]["hidden_dim"],
        head_layers=model_config["head"][task]["num_layers"],
        dropout=model_config["head"][task]["dropout"],
        task_level="node",
    ).to(device)


def load_ensemble(task, model_config, num_classes, active_levels, readout_level,
                   device, k_folds, seed, ckpt_path=None):
    level_tag = "_".join(active_levels)
    seed_tag = f"seed{seed}"

    ckpt_paths = []
    if ckpt_path is not None:
        ckpt_paths = [ckpt_path]
    elif k_folds > 1:
        for fold_idx in range(k_folds):
            p = f"./ckpts/best_prime_{task}_{level_tag}_{seed_tag}_fold{fold_idx}.pt"
            if not os.path.exists(p):
                raise ValueError(f"Fold checkpoint not found: {p}")
            ckpt_paths.append(p)
    else:
        p = f"./ckpts/best_prime_{task}_{level_tag}_{seed_tag}.pt"
        if not os.path.exists(p):
            raise ValueError(f"Checkpoint not found: {p}")
        ckpt_paths = [p]

    print(f"    Loading {len(ckpt_paths)} checkpoint(s) for task '{task}': {ckpt_paths}")
    models = []
    for p in ckpt_paths:
        m = build_model(model_config, task, num_classes, active_levels, readout_level, device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    return models


# ======================================================
# Chain-aware label matching (positions_by_chain -> residue_labels)
# ======================================================

def labels_from_positions_by_chain(residue_id, positions_by_chain):
    positive = set()
    for chain_id, resnums in positions_by_chain.items():
        for r in resnums:
            positive.add(f"{chain_id}:{r}")
    labels = [1 if rid in positive else 0 for rid in residue_id]
    n_pos = sum(labels)
    if n_pos == 0:
        print(f"    [warn] 0 of {sum(len(v) for v in positions_by_chain.values())} "
              f"labeled position(s) matched any residue_id in the graph -- "
              f"check chain letters match the clean PDB.")
    return torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def predict(models, graph, device, threshold, residue_labels):
    graph.to_torch(device)
    graph.residue_labels = residue_labels.to(device)

    all_probs = []
    for m in models:
        logits = m(graph)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs)
    probs = torch.stack(all_probs, dim=0).mean(dim=0).cpu().numpy()

    true_labels = graph.residue_labels.cpu().numpy()
    pred_labels = (probs >= threshold).astype(int)
    return true_labels, pred_labels, probs


def compute_metrics(true_labels, pred_labels):
    tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (true_labels == 0)))
    fp = int(np.sum((pred_labels == 1) & (true_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (true_labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def classify_residues(true_labels, pred_labels, residue_id):
    """{'chain:resnum': 'TP'|'TN'|'FP'|'FN'} -- chain-qualified, unlike
    render_prime_predictions.py's classify_residues, for the same reason
    labels_from_positions_by_chain exists (overlapping resnums across
    chains in these multi-chain targets)."""
    classification = {}
    for rid, t, p in zip(residue_id, true_labels, pred_labels):
        if t == 1 and p == 1:
            classification[rid] = "TP"
        elif t == 0 and p == 0:
            classification[rid] = "TN"
        elif t == 0 and p == 1:
            classification[rid] = "FP"
        else:
            classification[rid] = "FN"
    return classification


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True,
                         help="extract_case_study_labels.py's --pdb_dir "
                              "(contains {target}_clean.pdb)")
    parser.add_argument("--labels_dir", type=str, required=True,
                         help="extract_case_study_labels.py's --output_dir "
                              "(contains {target}.json)")
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--graph_dir", type=str, required=True,
                         help="Where to save {target}__{site}.pt graphs, "
                              "compatible with render_prime_predictions.py's --graph_dir.")
    parser.add_argument("--task_dir", type=str, required=True,
                         help="Where to copy PDBs/{target}__{site}.pdb, "
                              "compatible with render_prime_predictions.py's --task_dir.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Where to save per-(target,site) metrics/predictions JSON.")
    parser.add_argument("--atom_encoder_path", type=str, default=None)
    parser.add_argument("--surface_encoder_path", type=str, default=None)
    parser.add_argument("--active_levels", nargs="+",
                         default=["surface", "atom", "residue", "sse", "protein"])
    parser.add_argument("--readout_level", type=str, default="residue")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k_folds", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.graph_dir, exist_ok=True)
    os.makedirs(os.path.join(args.task_dir, "PDBs"), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    # ── optional shared encoders (same pattern as hierarchical_graph.py's __main__) ──
    atom_encoder = None
    surface_encoder = None
    if args.atom_encoder_path:
        from models.egnn import EGNN
        print(f"Loading atom encoder from {args.atom_encoder_path}")
        ckpt = torch.load(args.atom_encoder_path, map_location="cpu")
        cfg = ckpt["config"]["atom"]
        atom_encoder = EGNN(in_node_nf=cfg["atom_feat_dim"], hidden_nf=cfg["hidden_dim"],
                             out_node_nf=cfg["latent_dim"], in_edge_nf=1, n_layers=cfg["n_layers"])
        atom_encoder.load_state_dict(ckpt["model_state_dict"])
        atom_encoder = atom_encoder.to(device).eval()
    if args.surface_encoder_path:
        from models.emnn import EMNN
        print(f"Loading surface encoder from {args.surface_encoder_path}")
        ckpt = torch.load(args.surface_encoder_path, map_location="cpu")
        cfg = ckpt["config"]["surface"]
        surface_encoder = EMNN(in_node_nf=cfg["input_dim"], hidden_nf=cfg["hidden_dim"],
                                out_node_nf=cfg["latent_dim"], in_edge_nf=1, n_layers=cfg["n_layers"])
        surface_encoder.load_state_dict(ckpt["model_state_dict"])
        surface_encoder = surface_encoder.to(device).eval()

    # cache loaded checkpoint ensembles across targets/sites (e.g. SMPI-site
    # is reused by both thrombin and er_alpha -- load it once)
    model_cache = {}

    def get_models(task):
        if task not in model_cache:
            task_cfg = data_config["tasks"][task]
            model_cache[task] = load_ensemble(
                task, model_config, task_cfg["num_classes"],
                args.active_levels, args.readout_level, device,
                args.k_folds, args.seed,
            )
        return model_cache[task]

    clean_pdbs = sorted(glob.glob(os.path.join(args.pdb_dir, "*_clean.pdb")))
    if not clean_pdbs:
        raise ValueError(f"No *_clean.pdb found in {args.pdb_dir} -- "
                          f"run extract_case_study_labels.py first.")

    all_metrics = {}

    for pdb_path in clean_pdbs:
        target_name = os.path.basename(pdb_path)[: -len("_clean.pdb")]
        if target_name not in CASE_STUDY_SITE_TASKS:
            print(f"[skip] {target_name}: not in CASE_STUDY_SITE_TASKS, add it there first.")
            continue

        labels_path = os.path.join(args.labels_dir, f"{target_name}.json")
        if not os.path.exists(labels_path):
            print(f"[skip] {target_name}: {labels_path} not found.")
            continue
        with open(labels_path) as f:
            site_labels = json.load(f)

        print(f"\n=== {target_name} ===")
        print(f"  Building graph from {pdb_path} ...")
        process_data = load_protein_from_pdb(pdb_path)
        surface_tmp = os.path.join(args.graph_dir, f"_tmp_{target_name}.obj")
        export_surface(pdb_path=pdb_path, output_path=surface_tmp,
                        surface_quality=0, solvent_radius=1.4, selection="all")
        try:
            graph = build_hierarchical_protein_graph(
                surface_path=surface_tmp,
                process_data=process_data,
                pdb_path=pdb_path,
                residue_labels=None,
                atom_encoder=atom_encoder,
                surface_encoder=surface_encoder,
                device=device,
            )
        finally:
            if os.path.exists(surface_tmp):
                os.remove(surface_tmp)

        for site_name, task in CASE_STUDY_SITE_TASKS[target_name].items():
            if site_name not in site_labels:
                print(f"  [skip] {target_name}/{site_name}: not found in {labels_path} "
                      f"(available: {list(site_labels.keys())})")
                continue

            print(f"  {site_name} (task='{task}'):")
            models = get_models(task)
            residue_labels = labels_from_positions_by_chain(
                graph.residue_id, site_labels[site_name]["positions_by_chain"]
            )

            true_labels, pred_labels, probs = predict(
                models, graph, device, args.threshold, residue_labels
            )
            metrics = compute_metrics(true_labels, pred_labels)
            print(f"    F1={metrics['f1']:.3f}  Precision={metrics['precision']:.3f}  "
                  f"Recall={metrics['recall']:.3f}  MCC={metrics['mcc']:.3f}  "
                  f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']})")

            record_id = f"{target_name}__{site_name}"
            classification = classify_residues(true_labels, pred_labels, graph.residue_id)

            # save graph (with this site's residue_labels baked in) + PDB copy,
            # ready for render_prime_predictions.py --record_id <record_id>
            graph.residue_labels = residue_labels
            torch.save(graph, os.path.join(args.graph_dir, f"{record_id}.pt"))
            shutil.copy(pdb_path, os.path.join(args.task_dir, "PDBs", f"{record_id}.pdb"))

            out_path = os.path.join(args.output_dir, f"{record_id}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "record_id": record_id,
                    "task": task,
                    "pdb_source": site_labels[site_name]["source"],
                    **metrics,
                    "residue_classification": classification,
                }, f, indent=2)
            print(f"    Saved: {out_path}")
            all_metrics[record_id] = metrics

    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary: {summary_path}")
    print("Render any case with your existing pipeline, e.g.:")
    print(f"  python render_prime_predictions.py --record_id <target>__<site> "
          f"--graph_dir {args.graph_dir} --task_dir {args.task_dir} --output_dir <figs_dir>")


if __name__ == "__main__":
    main()