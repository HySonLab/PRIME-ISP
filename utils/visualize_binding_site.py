import argparse
import json
import os
import sys
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.models import PRIME
from utils.helpers import load_config
from utils.label_formats import load_ion_positions
from utils.hierarchical_graph import *

# ======================================================
# Color scheme -- matches the reference figure
# ======================================================
COLORS = {
    "TP": "#E8837A",
    "TN": "#D9D9D9",
    "FP": "#7FB8D9",
    "FN": "#8E3FA8",
}
COLORS_PYMOL = {  # PyMOL wants RGB 0-1 tuples
    "TP": (0.910, 0.514, 0.478),
    "TN": (0.851, 0.851, 0.851),
    "FP": (0.498, 0.722, 0.851),
    "FN": (0.557, 0.247, 0.659),
}

VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}


# ======================================================
# Model loading (mirrors test_prime.py's ensemble pattern)
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
                   device, k_folds, seed):
    level_tag = "_".join(active_levels)
    seed_tag = f"seed{seed}"

    ckpt_paths = []
    if k_folds > 1:
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

    print(f"Loading {len(ckpt_paths)} checkpoint(s) for '{task}': {ckpt_paths}")
    models = []
    for p in ckpt_paths:
        m = build_model(model_config, task, num_classes, active_levels, readout_level, device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    return models


def labels_from_positions(residue_id, positions):
    labels = []
    for rid in residue_id:
        try:
            res_num = int(rid.split(":")[1])
        except (IndexError, ValueError):
            labels.append(0)
            continue
        labels.append(1 if res_num in positions else 0)
    return torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def predict_one(models, graph, device, threshold, label_positions=None, record_id=None):
    graph.to_torch(device)

    if graph.residue_labels is None:
        if label_positions is None or record_id not in label_positions:
            raise ValueError(f"No true labels available for {record_id}")
        graph.residue_labels = labels_from_positions(graph.residue_id, label_positions[record_id])

    all_probs = []
    for m in models:
        logits = m(graph)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs)
    probs = torch.stack(all_probs, dim=0).mean(dim=0).cpu().numpy()

    true_labels = graph.residue_labels.cpu().numpy()
    pred_labels = (probs >= threshold).astype(int)

    return true_labels, pred_labels, graph.residue_id


def classify_residues(true_labels, pred_labels, residue_id):
    """Returns {'chain:resnum': 'TP'|'TN'|'FP'|'FN'}, keyed by the FULL
    residue_id string, not resnum alone. resnum-only keys silently collide
    whenever a target has multiple chains with overlapping auth_seq_id
    ranges (e.g. case-study targets like thrombin's heavy+light chains) --
    two different residues would share one dict entry, and PyMOL's
    unqualified `resi N` selector would then color BOTH chains' residue N
    identically. Chain-qualifying is a no-op for single-chain tasks, so
    this is safe for every existing (non-case-study) task too."""
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


def _parse_chain_resnum(rid):
    chain_id, resnum = rid.split(":")
    return chain_id, int(resnum)


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


def find_pdb_path(task_dir, record_id):
    flat = os.path.join(task_dir, "PDBs", f"{record_id}.pdb")
    receptor = os.path.join(task_dir, "PDBs", "receptor", f"{record_id}.pdb")
    if os.path.exists(flat):
        return flat
    if os.path.exists(receptor):
        return receptor
    raise FileNotFoundError(f"No PDB found for {record_id} under {task_dir}/PDBs/")


# ======================================================
# Best/worst-case scan (by MCC)
# ======================================================

def scan_best_worst(models, test_ids, graph_dir, device, threshold, label_positions):
    """Scans every test protein, returns (best_id, best_metrics, worst_id,
    worst_metrics). Skips proteins with zero positive residues (MCC
    undefined) or missing graph files."""
    results = []
    for record_id in test_ids:
        graph_path = os.path.join(graph_dir, f"{record_id}.pt")
        if not os.path.exists(graph_path):
            continue
        try:
            graph = torch.load(graph_path, map_location=device, weights_only=False)
            true_labels, pred_labels, _ = predict_one(
                models, graph, device, threshold, label_positions, record_id
            )
            if true_labels.sum() == 0:
                continue
            metrics = compute_metrics(true_labels, pred_labels)
            results.append((record_id, metrics))
        except Exception as e:
            print(f"[WARN] skipped {record_id}: {e}")

    if not results:
        raise ValueError("No valid test proteins found to score.")

    results.sort(key=lambda r: r[1]["mcc"])
    worst_id, worst_metrics = results[0]
    best_id, best_metrics = results[-1]

    print(f"\nScanned {len(results)} test protein(s).")
    print(f"  Worst case: {worst_id}  MCC={worst_metrics['mcc']:.3f}")
    print(f"  Best case:  {best_id}  MCC={best_metrics['mcc']:.3f}")

    return best_id, best_metrics, worst_id, worst_metrics


# ======================================================
# Backend 1: pure Python (marching cubes + matplotlib)
# ======================================================

def load_atoms(pdb_path, chain_id=None):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    coords, elements, res_keys = [], [], []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[0] != " ":
                continue
            for atom in res:
                elem = atom.element.strip() if atom.element else atom.get_name()[0]
                if elem == "H":
                    continue
                coords.append(atom.get_coord())
                elements.append(elem)
                # chain-qualified, e.g. "H:20" -- matches classify_residues'
                # keys and avoids resnum collisions across a target's own
                # multiple chains (e.g. thrombin H/L overlapping numbering).
                res_keys.append(f"{chain.id}:{res.id[1]}")
    return np.array(coords, dtype=np.float32), elements, np.array(res_keys)


def build_density_surface(coords, elements, grid_spacing=0.7, padding=4.0, iso_level=0.55):
    from skimage import measure
    radii = np.array([VDW_RADII.get(e, 1.7) for e in elements], dtype=np.float32)

    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding
    dims = np.ceil((maxs - mins) / grid_spacing).astype(int)

    xs = np.linspace(mins[0], maxs[0], dims[0])
    ys = np.linspace(mins[1], maxs[1], dims[1])
    zs = np.linspace(mins[2], maxs[2], dims[2])

    density = np.zeros(dims, dtype=np.float32)

    for (x, y, z), r in zip(coords, radii):
        sigma = r * 0.9
        win = int(np.ceil(3 * sigma / grid_spacing)) + 1
        ix, iy, iz = (int((x - mins[0]) / grid_spacing),
                      int((y - mins[1]) / grid_spacing),
                      int((z - mins[2]) / grid_spacing))
        x0, x1 = max(0, ix - win), min(dims[0], ix + win)
        y0, y1 = max(0, iy - win), min(dims[1], iy + win)
        z0, z1 = max(0, iz - win), min(dims[2], iz + win)
        gx, gy, gz = np.meshgrid(xs[x0:x1], ys[y0:y1], zs[z0:z1], indexing="ij")
        d2 = (gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2
        density[x0:x1, y0:y1, z0:z1] += np.exp(-d2 / (2 * sigma ** 2))

    verts, faces, normals, _ = measure.marching_cubes(density, level=iso_level)
    verts = verts * grid_spacing + mins
    return verts, faces


def color_faces_by_residue(verts, faces, res_coords, res_keys, res_class):
    from scipy.spatial import cKDTree
    tree = cKDTree(res_coords)
    face_centroids = verts[faces].mean(axis=1)
    _, nearest_idx = tree.query(face_centroids)
    nearest_key = res_keys[nearest_idx]
    return [COLORS[res_class.get(k, "TN")] for k in nearest_key]


def render_matplotlib(pdb_path, chain, res_class, output_prefix, grid_spacing=0.7):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords, elements, res_keys_per_atom = load_atoms(pdb_path, chain)
    unique_keys = np.unique(res_keys_per_atom)
    res_coords = np.array([coords[res_keys_per_atom == k].mean(axis=0) for k in unique_keys])

    verts, faces = build_density_surface(coords, elements, grid_spacing=grid_spacing)
    face_colors = color_faces_by_residue(verts, faces, res_coords, unique_keys, res_class)

    for azim, suffix in [(-60, "front"), (120, "back")]:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces], facecolor=face_colors, edgecolor="none")
        ax.add_collection3d(mesh)
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=azim)
        ax.set_axis_off()
        out_path = f"{output_prefix}_{suffix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ======================================================
# Backend 2: PyMOL (optional, higher quality)
# ======================================================

def render_pymol(pdb_path, chain, res_class, output_prefix, zoom_buffer=8,
                  zoom_mode="whole"):
    try:
        import pymol2
    except ImportError:
        raise ImportError(
            "PyMOL not found. Install with: conda install -c conda-forge "
            "pymol-open-source   -- or use --backend matplotlib instead."
        )

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(pdb_path, "complex")
        if chain:
            cmd.remove(f"not chain {chain}")
        cmd.remove("solvent")
        cmd.hide("everything")
        cmd.show("surface")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 0)
        cmd.color("gray80", "all")

        # res_class is keyed by full "chain:resnum" residue_id -- always
        # color WITH a chain qualifier, since an unqualified `resi N` hits
        # every chain's residue N at once. For single-chain targets this is
        # a no-op filter; for multi-chain targets (e.g. thrombin H/L, whose
        # auth_seq_id ranges overlap) it's the difference between coloring
        # the right residue and silently coloring the wrong one too.
        for rid, label in res_class.items():
            chain_id, resnum = _parse_chain_resnum(rid)
            r, g, b = COLORS_PYMOL[label]
            color_name = f"col_{label}_{chain_id}_{resnum}"
            cmd.set_color(color_name, [r, g, b])
            cmd.color(color_name, f"chain {chain_id} and resi {resnum}")

        # Residues actually classified TP/FP/FN -- the "interesting" site,
        # as opposed to background TN. When the site is small relative to
        # the whole structure (e.g. a 3-5 residue metal-binding site on an
        # otherwise large protein), zooming to "all" leaves it a
        # barely-visible speck; zoom_mode="site" frames tightly on just
        # this region instead. Framing for every other task is unchanged
        # unless you explicitly pass --zoom_mode site for it too.
        site_rids = [rid for rid, label in res_class.items() if label != "TN"]
        use_site_zoom = (zoom_mode == "site" and len(site_rids) > 0)
        if zoom_mode == "site" and not site_rids:
            print("  Note: --zoom_mode site requested but no TP/FP/FN "
                  "residues were classified -- falling back to the "
                  "whole-structure zoom.")
        site_sel = None
        if site_rids:
            by_chain = {}
            for rid in site_rids:
                c, r = _parse_chain_resnum(rid)
                by_chain.setdefault(c, []).append(str(r))
            site_sel = " or ".join(
                f"(chain {c} and resi {'+'.join(rs)})" for c, rs in by_chain.items()
            )

        # Orient on the WHOLE structure even in site mode. cmd.orient() on
        # a tiny, tightly clustered selection (3-5 residues around one
        # metal ion) computes a degenerate PCA basis -- near-zero extent
        # along one axis -- and the camera distance PyMOL derives from
        # that ends up placing the camera essentially inside the surface
        # mesh (black cave-wall artifacts, nothing recognizable). Orienting
        # on the whole complex keeps a stable, sane camera direction; only
        # the ZOOM target (site vs. whole) needs to differ.
        cmd.orient()
        if use_site_zoom:
            # A concave binding pocket's actual surface can bulge closer to
            # the camera than the atom-center buffer accounts for, so site
            # zoom needs extra clearance beyond the buffer value used for
            # whole-structure shots to avoid grazing/clipping into the wall.
            site_zoom_buffer = max(zoom_buffer, 14)
            cmd.zoom(site_sel, buffer=site_zoom_buffer)
        else:
            cmd.zoom("all", buffer=zoom_buffer)
        cmd.ray(1200, 1200)
        cmd.png(f"{output_prefix}_front.png", dpi=300)
        print(f"Saved: {output_prefix}_front.png")

        cmd.turn("y", 180)
        if use_site_zoom:
            cmd.zoom(site_sel, buffer=site_zoom_buffer)
        else:
            cmd.zoom("all", buffer=zoom_buffer)
        cmd.ray(1200, 1200)
        cmd.png(f"{output_prefix}_back.png", dpi=300)
        print(f"Saved: {output_prefix}_back.png")


# ======================================================
# Visualize one protein end-to-end
# ======================================================

def visualize_one(models, record_id, graph_dir, task_dir, device, threshold,
                    label_positions, chain, backend, grid_spacing, zoom_buffer,
                    output_prefix, zoom_mode="whole"):
    graph_path = os.path.join(graph_dir, f"{record_id}.pt")
    graph = torch.load(graph_path, map_location=device, weights_only=False)

    true_labels, pred_labels, residue_id = predict_one(
        models, graph, device, threshold, label_positions, record_id
    )
    metrics = compute_metrics(true_labels, pred_labels)
    print(f"{record_id}: F1={metrics['f1']:.3f}  Precision={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}  MCC={metrics['mcc']:.3f}")

    # Export this specific rendered case's own metrics as JSON, alongside
    # the _front.png/_back.png -- make_qual_figure.py reads this to caption
    # the panel with the metrics of the actual protein pictured, instead
    # of the whole-test-set aggregate PRIME-ISP score from the main
    # results table (those two numbers can differ quite a bit, since the
    # pictured case is usually hand-picked as the best- or worst-scoring
    # one, not a "typical" protein).
    metrics_path = f"{output_prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"record_id": record_id, **metrics}, f, indent=2)
    print(f"Saved: {metrics_path}")

    res_class = classify_residues(true_labels, pred_labels, residue_id)
    pdb_path = find_pdb_path(task_dir, record_id)

    if backend == "pymol":
        render_pymol(pdb_path, chain, res_class, output_prefix, zoom_buffer, zoom_mode)
    else:
        if zoom_mode == "site":
            print("  Note: --zoom_mode site is only implemented for the "
                  "pymol backend; the matplotlib backend always frames the "
                  "whole structure. Use --backend pymol for a tight zoom.")
        render_matplotlib(pdb_path, chain, res_class, output_prefix, grid_spacing)

    return metrics

# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--record_id", type=str, default=None,
                         help="Visualize one specific protein instead of best/worst.")
    parser.add_argument("--n_samples", type=int, default=None,
                         help="Visualize N random proteins instead of best/worst.")
    parser.add_argument("--test_name", type=str, default=None,
                         help="For tasks with multiple named test sets.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chain", type=str, default=None)
    parser.add_argument("--backend", choices=["matplotlib", "pymol"], default="matplotlib")
    parser.add_argument("--zoom_buffer", type=float, default=8,
                         help="PyMOL backend only: extra margin (Angstrom) "
                              "around the structure/site so nothing gets cropped.")
    parser.add_argument("--zoom_mode", choices=["whole", "site"], default="whole",
                         help="PyMOL backend only. 'whole' (default) zooms to fit the "
                              "entire structure -- unchanged framing, same as before. "
                              "'site' zooms tight on just the TP/FP/FN residues instead; "
                              "use this for tasks where the binding site is small "
                              "relative to the whole protein (e.g. MPI-Ca/Fe/Mn), where "
                              "a whole-structure zoom makes the site nearly invisible.")
    parser.add_argument("--grid_spacing", type=float, default=0.7)
    parser.add_argument(
        "--active_levels", nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"]
    )
    parser.add_argument("--readout_level", type=str, default="residue")
    parser.add_argument("--output_dir", type=str, default="./outputs/viz")
    parser.add_argument("--graph_dir", type=str, default=None,
                         help="Override the task-config-derived graph directory "
                              "(e.g. for case-study record_ids saved by "
                              "run_case_study_inference.py). Defaults to "
                              "data_config.yaml's tasks.<task>.graph_dir.")
    parser.add_argument("--task_dir", type=str, default=None,
                         help="Override the task-config-derived task directory "
                              "(PDBs are read from <task_dir>/PDBs). Defaults to "
                              "the parent of tasks.<task>.graph_dir.")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    task_cfg = data_config["tasks"][args.task]
    num_classes = task_cfg["num_classes"]
    graph_dir = args.graph_dir if args.graph_dir else task_cfg["graph_dir"]
    task_dir = args.task_dir if args.task_dir else os.path.dirname(task_cfg["graph_dir"])
    # assumes graph_dir = {task_dir}/Graphs when not overridden

    label_positions = None
    if "label_files" in task_cfg:
        label_positions = load_ion_positions(task_cfg["label_files"])

    test_cfg = task_cfg["splits"]["test"]
    if isinstance(test_cfg, dict):
        test_cfg_str_keys = {str(k): v for k, v in test_cfg.items()}
        if args.test_name:
            if str(args.test_name) not in test_cfg_str_keys:
                raise ValueError(
                    f"--test_name '{args.test_name}' not found for task "
                    f"'{args.task}'. Available: {list(test_cfg.keys())}"
                )
            test_split = test_cfg_str_keys[str(args.test_name)]
        else:
            test_split = list(test_cfg_str_keys.values())[0]
    else:
        test_split = test_cfg

    with open(test_split) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    models = load_ensemble(args.task, model_config, num_classes, args.active_levels,
                            args.readout_level, device, args.k_folds, args.seed)

    if args.record_id:
        # ---- specific protein ----
        output_prefix = os.path.join(args.output_dir, args.record_id)
        visualize_one(models, args.record_id, graph_dir, task_dir, device, args.threshold,
                       label_positions, args.chain, args.backend, args.grid_spacing,
                       args.zoom_buffer, output_prefix, args.zoom_mode)

    elif args.n_samples:
        # ---- random sample ----
        chosen_ids = random.sample(test_ids, min(args.n_samples, len(test_ids)))
        print(f"Visualizing {len(chosen_ids)} random sample(s): {chosen_ids}")
        for record_id in chosen_ids:
            output_prefix = os.path.join(args.output_dir, record_id)
            visualize_one(models, record_id, graph_dir, task_dir, device, args.threshold,
                           label_positions, args.chain, args.backend, args.grid_spacing,
                           args.zoom_buffer, output_prefix, args.zoom_mode)

    else:
        # ---- default: find + render best/worst MCC case automatically ----
        best_id, best_metrics, worst_id, worst_metrics = scan_best_worst(
            models, test_ids, graph_dir, device, args.threshold, label_positions
        )

        print(f"\nRendering best case: {best_id}")
        best_prefix = os.path.join(args.output_dir, "best")
        visualize_one(models, best_id, graph_dir, task_dir, device, args.threshold,
                       label_positions, args.chain, args.backend, args.grid_spacing,
                       args.zoom_buffer, best_prefix, args.zoom_mode)

        print(f"\nRendering worst case: {worst_id}")
        worst_prefix = os.path.join(args.output_dir, "worst")
        visualize_one(models, worst_id, graph_dir, task_dir, device, args.threshold,
                       label_positions, args.chain, args.backend, args.grid_spacing,
                       args.zoom_buffer, worst_prefix, args.zoom_mode)