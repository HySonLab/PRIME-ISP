"""
visualize_binding_site.py

Self-contained: given a task name, loads the trained (k-fold ensemble)
PRIME-ISP model, picks one or more test proteins (random sample or a given
record id), runs prediction, classifies each residue as TP/TN/FP/FN against
the true labels, and renders a surface visualization colored accordingly --
matching the style of Figure 2b in the ALLSites paper.

Two rendering backends:
  --backend matplotlib (default) -- pure Python, marching cubes on a Gaussian
      atomic-density field, no external tools required.
  --backend pymol -- higher-quality ray-traced rendering via the PyMOL Python
      API, used automatically if available and requested; requires PyMOL
      installed separately (e.g. `conda install -c conda-forge pymol-open-source`).

Usage:
    # random sample from the test set
    python visualize_binding_site.py --task DPI-site --n_samples 1

    # a specific protein
    python visualize_binding_site.py --task DPI-site --record_id 5dv7_C

    # with PyMOL rendering, if installed
    python visualize_binding_site.py --task DPI-site --record_id 5dv7_C --backend pymol
"""

import argparse
import os
import sys
import random
import json
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
    """Returns {resnum: 'TP'|'TN'|'FP'|'FN'} using residue_id ('chain:resnum')."""
    classification = {}
    for rid, t, p in zip(residue_id, true_labels, pred_labels):
        try:
            resnum = int(rid.split(":")[1])
        except (IndexError, ValueError):
            continue
        if t == 1 and p == 1:
            classification[resnum] = "TP"
        elif t == 0 and p == 0:
            classification[resnum] = "TN"
        elif t == 0 and p == 1:
            classification[resnum] = "FP"
        else:
            classification[resnum] = "FN"
    return classification


def compute_metrics(true_labels, pred_labels):
    tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (true_labels == 0)))
    fp = int(np.sum((pred_labels == 1) & (true_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (true_labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "mcc": mcc}


def find_pdb_path(task_dir, record_id):
    flat = os.path.join(task_dir, "PDBs", f"{record_id}.pdb")
    receptor = os.path.join(task_dir, "PDBs", "receptor", f"{record_id}.pdb")
    if os.path.exists(flat):
        return flat
    if os.path.exists(receptor):
        return receptor
    raise FileNotFoundError(f"No PDB found for {record_id} under {task_dir}/PDBs/")


# ======================================================
# Backend 1: pure Python (marching cubes + matplotlib)
# ======================================================

def load_atoms(pdb_path, chain_id=None):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    coords, elements, res_nums = [], [], []
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
                res_nums.append(res.id[1])
    return np.array(coords, dtype=np.float32), elements, np.array(res_nums)


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


def color_faces_by_residue(verts, faces, res_coords, res_nums, res_class):
    from scipy.spatial import cKDTree
    tree = cKDTree(res_coords)
    face_centroids = verts[faces].mean(axis=1)
    _, nearest_idx = tree.query(face_centroids)
    nearest_resnum = res_nums[nearest_idx]
    return [COLORS[res_class.get(int(r), "TN")] for r in nearest_resnum]


def render_matplotlib(pdb_path, chain, res_class, output_prefix, grid_spacing=0.7):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords, elements, res_nums_per_atom = load_atoms(pdb_path, chain)
    unique_resnums = np.unique(res_nums_per_atom)
    res_coords = np.array([coords[res_nums_per_atom == r].mean(axis=0) for r in unique_resnums])

    verts, faces = build_density_surface(coords, elements, grid_spacing=grid_spacing)
    face_colors = color_faces_by_residue(verts, faces, res_coords, unique_resnums, res_class)

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

def render_pymol(pdb_path, chain, res_class, output_prefix, zoom_buffer=8):
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

        for resnum, label in res_class.items():
            r, g, b = COLORS_PYMOL[label]
            color_name = f"col_{label}_{resnum}"
            cmd.set_color(color_name, [r, g, b])
            cmd.color(color_name, f"resi {resnum}")

        cmd.orient()
        cmd.zoom("all", buffer=zoom_buffer)
        cmd.ray(1200, 1200)
        cmd.png(f"{output_prefix}_front.png", dpi=300)
        print(f"Saved: {output_prefix}_front.png")

        cmd.turn("y", 180)
        cmd.zoom("all", buffer=zoom_buffer)
        cmd.ray(1200, 1200)
        cmd.png(f"{output_prefix}_back.png", dpi=300)
        print(f"Saved: {output_prefix}_back.png")


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--record_id", type=str, default=None,
                         help="Specific protein to visualize. If omitted, samples "
                              "--n_samples random proteins from the test split.")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--test_name", type=str, default=None,
                         help="For tasks with multiple named test sets.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chain", type=str, default=None)
    parser.add_argument("--backend", choices=["matplotlib", "pymol"], default="matplotlib")
    parser.add_argument("--zoom_buffer", type=float, default=8,
                         help="PyMOL backend only: extra margin (Angstrom) "
                              "around the structure so nothing gets cropped.")
    parser.add_argument("--grid_spacing", type=float, default=0.7)
    parser.add_argument(
        "--active_levels", nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"]
    )
    parser.add_argument("--readout_level", type=str, default="residue")
    parser.add_argument("--output_dir", type=str, default="./outputs/viz")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    task_cfg = data_config["tasks"][args.task]
    num_classes = task_cfg["num_classes"]
    graph_dir = task_cfg["graph_dir"]
    task_dir = os.path.dirname(graph_dir)  # assumes graph_dir = {task_dir}/Graphs

    label_positions = None
    if "label_files" in task_cfg:
        label_positions = load_ion_positions(task_cfg["label_files"])

    test_cfg = task_cfg["splits"]["test"]
    if isinstance(test_cfg, dict):
        test_split = test_cfg[args.test_name] if args.test_name else list(test_cfg.values())[0]
    else:
        test_split = test_cfg
    with open(test_split) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    if args.record_id:
        chosen_ids = [args.record_id]
    else:
        chosen_ids = random.sample(test_ids, min(args.n_samples, len(test_ids)))
    print(f"Visualizing: {chosen_ids}")

    models = load_ensemble(args.task, model_config, num_classes, args.active_levels,
                            args.readout_level, device, args.k_folds, args.seed)

    for record_id in chosen_ids:
        print(f"\n{'='*40}\n{record_id}\n{'='*40}")
        graph_path = os.path.join(graph_dir, f"{record_id}.pt")
        graph = torch.load(graph_path, map_location=device, weights_only=False)

        true_labels, pred_labels, residue_id = predict_one(
            models, graph, device, args.threshold, label_positions, record_id
        )
        metrics = compute_metrics(true_labels, pred_labels)
        print(f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
              f"MCC={metrics['mcc']:.3f}")

        res_class = classify_residues(true_labels, pred_labels, residue_id)
        pdb_path = find_pdb_path(task_dir, record_id)
        output_prefix = os.path.join(args.output_dir, record_id)

        if args.backend == "pymol":
            render_pymol(pdb_path, args.chain, res_class, output_prefix, args.zoom_buffer)
        else:
            render_matplotlib(pdb_path, args.chain, res_class, output_prefix, args.grid_spacing)