"""
predict.py -- run PRIME-ISP on your own protein.

Input : an amino-acid sequence (string or FASTA) OR a structure (PDB / mmCIF),
        plus the interaction partner type(s) to predict.
Output: per-residue interaction probabilities (CSV) and, for each partner, a PDB
        file with the probability stored in the B-factor column (colour by
        B-factor in PyMOL / Mol* / py3Dmol to see the predicted site).

Examples
--------
    # from a sequence (folded with ESMFold2), all partner types
    python predict.py --sequence MKTAYIAKQRQISFVKSHFSRQ... --partners all

    # from a FASTA file with several proteins, selected partners
    python predict.py --fasta my_proteins.fasta --partners protein dna small_molecule

    # from a structure, only chain A
    python predict.py --structure my_protein.pdb --chains A --partners metal_ca peptide

Run from the repository root (needs ./ckpts, ./config/model_config.yaml, a GPU).
"""

import argparse
import csv
import gc
import os
import re
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ----------------------------------------------------------------------
# Partner type -> task (= which trained model ensemble is used).
# Edit here to choose a different benchmark model for a partner type.
# ----------------------------------------------------------------------
PARTNER_TASKS = {
    "protein":        "PPI-site_Train335-Test60-Test315",
    "peptide":        "PepPI-site_Train1154-Test125",
    "dna":            "DPI-site",
    "rna":            "RPI-site",
    "metal_ca":       "MPI-site_CA",
    "metal_fe":       "MPI-site_FE",
    "metal_mn":       "MPI-site_MN",
    "carbohydrate":   "CarbPI-site",
    "small_molecule": "SMPI-site",
}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ======================================================================
# Inputs
# ======================================================================

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "protein"


def read_sequences(sequence=None, fasta=None):
    """Return {name: sequence}."""
    seqs = {}
    if sequence:
        seqs["query"] = sequence
    if fasta:
        name = None
        with open(fasta) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    name = safe_name(line[1:].split()[0])
                    seqs[name] = ""
                elif name is not None:
                    seqs[name] += line
    clean = {}
    for name, seq in seqs.items():
        seq = re.sub(r"\s+", "", seq).upper()
        bad = set(seq) - VALID_AA
        if bad:
            raise ValueError(f"{name}: unsupported characters {sorted(bad)}. "
                             f"Only the 20 standard amino acids are supported.")
        if not seq:
            raise ValueError(f"{name}: empty sequence.")
        clean[name] = seq
    return clean


def fold_sequences(seqs, out_dir):
    """Fold sequences with ESMFold2 (same model/settings as folding.py)."""
    from folding import load_model, try_fold, free_gpu_memory

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    todo = {}
    for name, seq in seqs.items():
        path = os.path.join(out_dir, f"{name}.pdb")
        paths[name] = path
        if not os.path.exists(path):
            todo[name] = seq
    if todo:
        model = load_model()
        for name, seq in todo.items():
            print(f"  folding {name} ({len(seq)} residues) ...")
            pdb_str, _ = try_fold(model, seq, num_diffusion_samples=1)
            if pdb_str is None:
                raise RuntimeError(f"ESMFold2 ran out of memory on {name} "
                                   f"({len(seq)} residues).")
            with open(paths[name], "w") as f:
                f.write(pdb_str)
        del model
        free_gpu_memory()
    return paths


def clean_structure(path, out_path, chains=None):
    """Keep the first model, protein residues only (no waters / ligands /
    non-standard residues), optionally a subset of chains."""
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    parser = (MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    structure = parser.get_structure("query", path)
    model = next(iter(structure))
    wanted = set(chains) if chains else None

    class ProteinSelect(Select):
        def accept_model(self, m):
            return m.get_id() == model.get_id()

        def accept_chain(self, chain):
            return wanted is None or chain.get_id() in wanted

        def accept_residue(self, res):
            return res.get_id()[0] == " " and res.get_resname() in STANDARD_RESIDUES

        def accept_atom(self, atom):
            return atom.get_altloc() in (" ", "A")

    long_ids = [c.get_id() for c in model if len(c.get_id()) > 1]
    if long_ids:
        raise ValueError(f"Chain ids {long_ids} are longer than one character; "
                         f"please rename chains or convert to PDB format.")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, ProteinSelect())
    return out_path


# ======================================================================
# Graph + models
# ======================================================================

def build_graph(pdb_path, work_dir, device):
    from utils.export_surface import export_surface
    from utils.hierarchical_graph import (build_hierarchical_protein_graph,
                                          load_protein_from_pdb)

    process_data = load_protein_from_pdb(pdb_path)
    surface_tmp = os.path.join(work_dir, "_tmp_surface.obj")
    export_surface(pdb_path=pdb_path, output_path=surface_tmp,
                   surface_quality=0, solvent_radius=1.4, selection="all")
    try:
        graph = build_hierarchical_protein_graph(
            surface_path=surface_tmp,
            process_data=process_data,
            pdb_path=pdb_path,
            residue_labels=None,
            atom_encoder=None,
            surface_encoder=None,
            device=device,
        )
    finally:
        if os.path.exists(surface_tmp):
            os.remove(surface_tmp)
    return graph, process_data


def build_model(model_config, task, active_levels, readout_level, device):
    from models.models import PRIME
    return PRIME(
        num_classes=2,
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


def load_ensemble(task, model_config, args, device):
    level_tag = "_".join(args.active_levels)
    models = []
    for fold in range(args.k_folds):
        ckpt = os.path.join(
            args.ckpt_dir,
            f"best_prime_{task}_{level_tag}_seed{args.seed}_fold{fold}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        m = build_model(model_config, task, args.active_levels,
                        args.readout_level, device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_probs(models, graph):
    probs = [torch.softmax(m(graph), dim=-1)[:, 1] for m in models]
    return torch.stack(probs, dim=0).mean(dim=0).cpu().numpy()


# ======================================================================
# Outputs
# ======================================================================

def write_csv(path, residue_id, residues, probs_by_partner, threshold):
    partners = list(probs_by_partner)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["chain", "resnum", "resname"]
        for p in partners:
            header += [f"prob_{p}", f"site_{p}"]
        w.writerow(header)
        for i, rid in enumerate(residue_id):
            chain, resnum = rid.split(":")
            row = [chain, resnum, residues[i]]
            for p in partners:
                pr = float(probs_by_partner[p][i])
                row += [f"{pr:.4f}", int(pr >= threshold)]
            w.writerow(row)


def write_bfactor_pdb(pdb_path, out_path, residue_id, probs):
    """Copy of the structure with probability*100 in the B-factor column."""
    from Bio.PDB import PDBIO, PDBParser
    structure = PDBParser(QUIET=True).get_structure("q", pdb_path)
    lookup = {rid: float(p) * 100.0 for rid, p in zip(residue_id, probs)}
    for res in structure[0].get_residues():
        key = f"{res.get_parent().get_id()}:{res.get_id()[1]}"
        for atom in res:
            atom.set_bfactor(lookup.get(key, 0.0))
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def render_figure(pdb_path, residue_id, probs, out_path, title, threshold):
    """PNG with (left) the C-alpha trace colored by predicted probability and
    (right) the per-residue probability profile."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from Bio.PDB import PDBParser

    cmap = LinearSegmentedColormap.from_list(
        "prob", ["#1F5F5B", "#2C7873", "#4CA1A3", "#A9D6D2",
                 "#E3F2F1", "#F1C6BE", "#E8837A"])

    structure = PDBParser(QUIET=True).get_structure("q", pdb_path)
    ca = {}
    for res in structure[0].get_residues():
        if "CA" in res:
            ca[f"{res.get_parent().get_id()}:{res.get_id()[1]}"] = res["CA"].get_coord()

    idx = [i for i, rid in enumerate(residue_id) if rid in ca]
    xyz = np.array([ca[residue_id[i]] for i in idx])
    p = np.asarray(probs, dtype=float)[idx]
    chains = [residue_id[i].split(":")[0] for i in idx]

    fig = plt.figure(figsize=(11, 4.4), dpi=300)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    start = 0
    for end in range(1, len(chains) + 1):
        if end == len(chains) or chains[end] != chains[start]:
            seg = xyz[start:end]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#BFC7C7", lw=0.8, zorder=1)
            start = end
    order = np.argsort(p)  # draw high-probability residues last
    sc = ax.scatter(xyz[order, 0], xyz[order, 1], xyz[order, 2], c=p[order],
                    cmap=cmap, vmin=0, vmax=1, s=14 + 30 * (p[order] >= threshold),
                    edgecolors="none", depthshade=False, zorder=2)
    span = np.ptp(xyz, axis=0)
    ax.set_box_aspect(span / span.max())
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Interaction probability", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(len(p)), p, width=1.0, color=cmap(p), linewidth=0)
    ax2.axhline(threshold, color="#4A5757", ls="--", lw=0.9)
    ax2.set_xlim(0, len(p))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Residue index", fontsize=9)
    ax2.set_ylabel("Interaction probability", fontsize=9)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


PROB_COLORS = ["#1F5F5B", "#2C7873", "#4CA1A3", "#A9D6D2",
               "#E3F2F1", "#F1C6BE", "#E8837A"]  # low -> high probability


def render_pymol(bfactor_pdb, out_prefix, views=("front", "back")):
    """Surface rendering colored by predicted probability. Expects the PDB
    written by write_bfactor_pdb (probability * 100 in the B-factor column).
    Requires PyMOL: conda install -c conda-forge pymol-open-source"""
    import pymol2

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(bfactor_pdb, "prot")
        cmd.hide("everything")
        cmd.show("surface")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.set("surface_quality", 1)
        cmd.set("ray_trace_mode", 1)

        names = []
        for i, hex_color in enumerate(PROB_COLORS):
            rgb = [int(hex_color[j:j + 2], 16) / 255.0 for j in (1, 3, 5)]
            cmd.set_color(f"prob{i}", rgb)
            names.append(f"prob{i}")
        cmd.spectrum("b", " ".join(names), "prot", minimum=0, maximum=100)

        cmd.orient()
        cmd.zoom("all", buffer=4)
        for view in views:
            if view == "back":
                cmd.turn("y", 180)
            cmd.ray(1400, 1400)
            cmd.png(f"{out_prefix}_{view}.png", dpi=300)


PARTNER_LABELS = {
    "protein": "Protein", "peptide": "Peptide", "dna": "DNA", "rna": "RNA",
    "metal_ca": "Metal ion (Ca)", "metal_fe": "Metal ion (Fe)",
    "metal_mn": "Metal ion (Mn)", "carbohydrate": "Carbohydrate",
    "small_molecule": "Small molecule",
}


def combine_figure(panels, out_path, title, ncols=3):
    """One figure with every partner's PyMOL rendering and a shared
    probability color bar. panels = [(label, png_path), ...]."""
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("prob", PROB_COLORS)

    def load_cropped(path, margin=12):
        img = plt.imread(path)[..., :3]
        mask = (img < 0.97).any(axis=-1)
        if not mask.any():
            return img
        ys, xs = np.where(mask)
        return img[max(ys.min() - margin, 0):ys.max() + margin,
                   max(xs.min() - margin, 0):xs.max() + margin]

    n = len(panels)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows + 1.0),
                             dpi=300, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, path) in zip(axes.ravel(), panels):
        ax.imshow(load_cropped(path))
        ax.set_title(label, fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])

    cax = fig.add_axes([0.30, 0.045, 0.40, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Predicted interaction probability", fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input (give one)")
    src.add_argument("--sequence", type=str, help="Amino-acid sequence.")
    src.add_argument("--fasta", type=str, help="FASTA file (one protein per record).")
    src.add_argument("--structure", type=str, help="PDB or mmCIF file.")
    ap.add_argument("--chains", nargs="+", default=None,
                    help="Chains to keep from --structure (default: all).")
    ap.add_argument("--partners", nargs="+", default=["all"],
                    help=f"Partner type(s): {', '.join(PARTNER_TASKS)} or 'all'.")
    ap.add_argument("--output_dir", type=str, default="./prime_isp_output")
    ap.add_argument("--model_config", type=str, default="./config/model_config.yaml")
    ap.add_argument("--ckpt_dir", type=str, default="./ckpts")
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--active_levels", nargs="+",
                    default=["surface", "atom", "residue", "sse", "protein"])
    ap.add_argument("--readout_level", type=str, default="residue")
    ap.add_argument("--render", choices=["auto", "pymol", "matplotlib", "none"],
                    default="auto",
                    help="Figures: 'pymol' = surface colored by probability "
                         "(needs PyMOL), 'matplotlib' = backbone + probability "
                         "profile, 'auto' = pymol if installed, else matplotlib.")
    args = ap.parse_args()

    if sum(x is not None for x in (args.sequence, args.fasta, args.structure)) != 1:
        ap.error("Give exactly one of --sequence, --fasta, --structure.")

    partners = list(PARTNER_TASKS) if "all" in args.partners else args.partners
    unknown = [p for p in partners if p not in PARTNER_TASKS]
    if unknown:
        ap.error(f"Unknown partner type(s) {unknown}. Choose from {list(PARTNER_TASKS)}.")

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required (graph construction loads ESM-2 on GPU).")
    device = torch.device("cuda")

    from utils.helpers import load_config
    model_config = load_config(args.model_config)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. structures -------------------------------------------------
    structures = {}
    if args.structure:
        name = safe_name(os.path.splitext(os.path.basename(args.structure))[0])
        path = os.path.join(args.output_dir, name, f"{name}_clean.pdb")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        structures[name] = clean_structure(args.structure, path, args.chains)
    else:
        seqs = read_sequences(args.sequence, args.fasta)
        print("Folding with ESMFold2 ...")
        folded = fold_sequences(seqs, os.path.join(args.output_dir, "structures"))
        for name, path in folded.items():
            d = os.path.join(args.output_dir, name)
            os.makedirs(d, exist_ok=True)
            structures[name] = clean_structure(path, os.path.join(d, f"{name}_clean.pdb"))

    # ---- 2. graphs (one per protein, shared by all partner types) ------
    print("Building hierarchical graphs ...")
    graphs = {}
    for name, path in structures.items():
        print(f"  {name}")
        graph, pd = build_graph(path, os.path.dirname(path), device)
        graph.to_torch(device)
        graphs[name] = (graph, list(pd.residues))

    # ---- 3. predict: one ensemble per partner type, loaded once --------
    probs = {name: {} for name in graphs}
    for partner in partners:
        task = PARTNER_TASKS[partner]
        print(f"Predicting '{partner}' (task {task}) ...")
        models = load_ensemble(task, model_config, args, device)
        for name, (graph, _) in graphs.items():
            probs[name][partner] = predict_probs(models, graph)
        del models
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 4. write results ----------------------------------------------
    for name, (graph, residues) in graphs.items():
        d = os.path.join(args.output_dir, name)
        write_csv(os.path.join(d, f"{name}_predictions.csv"),
                  graph.residue_id, residues, probs[name], args.threshold)
        panels = []
        for partner, p in probs[name].items():
            write_bfactor_pdb(structures[name],
                              os.path.join(d, f"{name}_{partner}.pdb"),
                              graph.residue_id, p)
            backend = args.render
            if backend == "auto":
                try:
                    import pymol2  # noqa: F401
                    backend = "pymol"
                except ImportError:
                    backend = "matplotlib"
            if backend == "pymol":
                views_dir = os.path.join(d, "views")
                os.makedirs(views_dir, exist_ok=True)
                prefix = os.path.join(views_dir, f"{name}_{partner}")
                render_pymol(os.path.join(d, f"{name}_{partner}.pdb"), prefix,
                             views=("front",))
                panels.append((PARTNER_LABELS.get(partner, partner),
                               f"{prefix}_front.png"))
            elif backend == "matplotlib":
                render_figure(structures[name], graph.residue_id, p,
                              os.path.join(d, f"{name}_{partner}.png"),
                              f"{name}: {partner}", args.threshold)
            n_sites = int((p >= args.threshold).sum())
            print(f"  {name} / {partner}: {n_sites} of {len(p)} residues "
                  f"predicted as interacting (p >= {args.threshold})")
        if panels:
            combine_figure(panels, os.path.join(d, f"{name}_summary.png"), name)
            print(f"  summary figure: {os.path.join(d, f'{name}_summary.png')}")
        print(f"  results: {d}")


if __name__ == "__main__":
    main()