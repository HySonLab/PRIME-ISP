from dataclasses import dataclass
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
import glob
import trimesh
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.partition import extract_partition_matrices, mesh_simplification_quadric_decimation, get_atom_coords
from utils.export_surface import export_surface
from models.egnn import EGNN
from models.emnn import EMNN
from torch_cluster import knn_graph
from torch_geometric.data import Data
from scipy.sparse import diags
from scipy import spatial
from esm import pretrained

# Load ESM-2 model once
esm2_model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
esm2_model = esm2_model.eval().to('cuda')
batch_converter = alphabet.get_batch_converter()

MAX_FACES = 1024

ATOM37_NAMES = [
    "N", "CA", "C", "O", "CB",
    "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2", "CE3",
    "CZ", "CZ2", "CZ3",
    "CH2",
    "ND1", "ND2", "NE", "NE1", "NE2",
    "NH1", "NH2",
    "NZ",
    "OD1", "OD2",
    "OE1", "OE2",
    "OG", "OG1",
    "OH",
    "SD",
    "SG",
    "OXT"
]

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_LIST    = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_LIST)}

# ── Chemical lookup tables ────────────────────────
HYDROPHOBICITY = {
    'ALA': 1.8,  'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5,  'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5,  'LEU': 3.8,  'LYS': -3.9,
    'MET': 1.9,  'PHE': 2.8,  'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}

HBOND_DONOR = {
    'N', 'NH1', 'NH2', 'NZ', 'ND1', 'ND2', 'NE',
    'NE1', 'NE2', 'OG', 'OG1', 'OH', 'NH'
}

HBOND_ACCEPTOR = {
    'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1',
    'OH', 'ND1', 'NE2', 'SD'
}

AROMATIC_RES   = {'PHE', 'TYR', 'TRP', 'HIS'}
CHARGED_POS    = {'ARG', 'LYS'}
CHARGED_NEG    = {'ASP', 'GLU'}

ELEMENT_RADII = {
    "C": 1.908,
    "N": 1.824,
    "O": 1.661,
    "S": 2.000,
    "P": 2.100,
}

# ======================================================
# Extended residue features
# ======================================================

# [solvent_accessibility, hydrophobicity, hydrophilicity, volume,
#  isoelectric_point, helix_pref, sheet_pref, coil_pref, charge, atom_count]
PHYSCHEM_TABLE = {
    "ALA": [0.11, 1.80, -0.5, 88.6, 6.00, 1.42, 0.83, 0.80,  0.0,  5],
    "ARG": [0.67, -4.5, 3.0, 173.4, 10.76, 0.98, 0.93, 0.90, 1.0, 11],
    "ASN": [0.43, -3.5, 0.2, 114.1, 5.41, 0.67, 0.89, 1.34,  0.0,  8],
    "ASP": [0.43, -3.5, 3.0, 111.1, 2.77, 1.01, 0.54, 1.35, -1.0,  8],
    "CYS": [0.17, 2.5, -1.0, 108.5, 5.07, 0.70, 1.19, 1.06,  0.0,  6],
    "GLN": [0.51, -3.5, 0.2, 143.8, 5.65, 1.11, 1.10, 0.84,  0.0,  9],
    "GLU": [0.50, -3.5, 3.0, 138.4, 3.22, 1.51, 0.37, 0.74, -1.0,  9],
    "GLY": [0.36, -0.4, 0.0, 60.1, 5.97, 0.57, 0.75, 1.56,  0.0,  4],
    "HIS": [0.36, -3.2, -0.5, 153.2, 7.59, 1.00, 0.87, 0.95,  0.5, 10],
    "ILE": [0.13, 4.5, -1.8, 166.7, 6.02, 1.08, 1.60, 0.47,  0.0,  8],
    "LEU": [0.16, 3.8, -1.8, 166.7, 5.98, 1.21, 1.30, 0.59,  0.0,  8],
    "LYS": [0.59, -3.9, 3.0, 168.6, 9.74, 1.16, 0.74, 1.01,  1.0,  9],
    "MET": [0.14, 1.9, -1.3, 162.9, 5.74, 1.45, 1.05, 0.60,  0.0,  8],
    "PHE": [0.10, 2.8, -2.5, 189.9, 5.48, 1.13, 1.38, 0.59,  0.0, 11],
    "PRO": [0.36, -1.6, 0.0, 112.7, 6.30, 0.57, 0.55, 1.52,  0.0,  7],
    "SER": [0.34, -0.8, 0.3, 89.0, 5.68, 0.77, 0.75, 1.43,  0.0,  6],
    "THR": [0.29, -0.7, -0.4, 116.1, 5.66, 0.83, 1.19, 0.96,  0.0,  7],
    "TRP": [0.10, -0.9, -3.4, 227.8, 5.89, 1.08, 1.37, 0.61,  0.0, 14],
    "TYR": [0.21, -1.3, -2.3, 193.6, 5.66, 0.69, 1.47, 1.14,  0.0, 12],
    "VAL": [0.14, 4.2, -1.5, 140.0, 5.96, 1.06, 1.70, 0.50,  0.0,  7],
}
PHYSCHEM_DEFAULT = list(np.mean(list(PHYSCHEM_TABLE.values()), axis=0))


def get_physchem_features(residues):
    """residues: list of 3-letter residue codes -> (N_res, 10) np.array,
    z-score normalized across the protein."""
    feats = np.array(
        [PHYSCHEM_TABLE.get(r.upper(), PHYSCHEM_DEFAULT) for r in residues],
        dtype=np.float32,
    )
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    return feats


# ── phi/psi + RSA (no external binary) ──────────────────────────────

MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
MAX_ASA_DEFAULT = float(np.mean(list(MAX_ASA.values())))


def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    v = b0 - (b0 * b1_norm).sum(-1, keepdims=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(-1, keepdims=True) * b1_norm
    x = (v * w).sum(-1)
    y = (np.cross(b1_norm, v) * w).sum(-1)
    return np.arctan2(y, x)


def get_residue_phi_psi(process_data):
    coords_np = process_data.coords.numpy()
    n_res = coords_np.shape[0]

    N_coord  = coords_np[:, 0, :]
    CA_coord = coords_np[:, 1, :]
    C_coord  = coords_np[:, 2, :]

    phi = np.zeros(n_res, dtype=np.float32)
    psi = np.zeros(n_res, dtype=np.float32)

    if n_res >= 2:
        phi[1:] = _dihedral(C_coord[:-1], N_coord[1:], CA_coord[1:], C_coord[1:])
        psi[:-1] = _dihedral(N_coord[:-1], CA_coord[:-1], C_coord[:-1], N_coord[1:])

    return np.stack(
        [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=1
    ).astype(np.float32)


def get_residue_rsa(process_data, pdb_path):
    n_res = process_data.coords.shape[0]

    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model     = structure[0]

        sr = ShrakeRupley()
        sr.compute(model, level="R")

        rsa = np.zeros(n_res, dtype=np.float32)
        for i, rid in enumerate(process_data.residue_id):
            ch, res_num = rid.split(":")
            res_num = int(res_num)
            res_name = process_data.residues[i]

            try:
                residue = model[ch][(" ", res_num, " ")]
                sasa = residue.sasa
            except (KeyError, AttributeError):
                sasa = 0.0

            max_asa = MAX_ASA.get(res_name.upper(), MAX_ASA_DEFAULT)
            rsa[i] = min(sasa / max_asa, 1.5)

        return rsa.reshape(-1, 1)

    except Exception as e:
        print(f"[WARN] SASA/RSA computation failed for {pdb_path}: {e}. "
              f"Falling back to zeros.")
        return np.zeros((n_res, 1), dtype=np.float32)


def get_residue_structural_features(process_data, pdb_path):
    phi_psi = get_residue_phi_psi(process_data)
    rsa     = get_residue_rsa(process_data, pdb_path)
    return np.concatenate([phi_psi, rsa], axis=1).astype(np.float32)


def get_per_residue_esm_features(sequence, esm2_model, batch_converter, device="cuda", layer=33):
    """Returns (N_res, esm_dim) per-residue embeddings, excluding the
    BOS/EOS special tokens ESM-2 adds automatically."""
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = esm2_model(tokens, repr_layers=[layer], return_contacts=False)

    reps = results["representations"][layer][0, 1:-1, :]
    return reps.cpu().numpy().astype(np.float32)


# ======================================================
# Utility functions
# ======================================================

def get_full_atom_coords(coords: torch.Tensor, fill_value: float = 1e-5):
    filled        = coords[:, :, 0] != fill_value
    nz            = filled.nonzero()
    residue_index = nz[:, 0]
    atom_type     = nz[:, 1]
    flat_coords   = coords.reshape(-1, 3)
    flat_coords   = flat_coords[coords.reshape(-1, 3)[:, 0] != fill_value]
    return flat_coords, residue_index, atom_type


def normalize_coords(x, mode="std", eps=1e-6):
    x = x - x.mean(dim=0, keepdim=True)
    if mode == "std":
        std = x.std(dim=0, keepdim=True)
        std[std < eps] = 1.0
        x = x / std
    elif mode == "radius":
        radius = torch.norm(x, dim=1).max()
        x = x / (radius + eps)
    return x


def normalize_adjacency(A):
    A = A.astype(np.float32)
    A.eliminate_zeros()
    degrees     = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt  = np.zeros_like(degrees)
    mask        = degrees > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    D_inv_sqrt  = diags(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def coarsen_adjacency(A_fine, Pi, normalize=False):
    A_coarse = Pi.T @ A_fine @ Pi
    if normalize:
        sizes = np.array(Pi.sum(axis=0)).flatten()
        inv   = np.reciprocal(sizes, where=sizes > 0)
        D     = csr_matrix(np.diag(inv))
        A_coarse = D @ A_coarse @ D
    return A_coarse


def build_knn_adjacency(coords: np.ndarray, k: int = 8) -> csr_matrix:
    N          = len(coords)
    coords_t   = torch.tensor(coords, dtype=torch.float32)
    edge_index = knn_graph(coords_t, k=k, loop=False)
    row        = edge_index[0].numpy()
    col        = edge_index[1].numpy()
    data       = np.ones(len(row), dtype=np.float32)
    A          = csr_matrix((data, (row, col)), shape=(N, N))
    A          = A + A.T
    A.data[:]  = 1.0
    return A


def build_surface_knn_adjacency(face_centroids: np.ndarray, k: int = 8) -> csr_matrix:
    centroids_t = torch.tensor(face_centroids, dtype=torch.float32)
    edge_index  = knn_graph(centroids_t, k=k, loop=False)
    row         = edge_index[0].numpy()
    col         = edge_index[1].numpy()
    data        = np.ones(len(row), dtype=np.float32)
    A           = csr_matrix(
        (data, (row, col)),
        shape=(len(face_centroids), len(face_centroids))
    )
    A          = A + A.T
    A.data[:]  = 1.0
    return A


# ======================================================
# PDB loading
# ======================================================

def load_protein_from_pdb(pdb_path: str):
    import warnings
    warnings.filterwarnings("ignore")

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model     = structure[0]

    coords_list     = []
    residues_list   = []
    residue_id_list = []
    chains_list     = []

    chain_map = {}
    chain_idx = 0

    for chain in model.get_chains():
        chain_id = chain.get_id()
        if chain_id not in chain_map:
            chain_map[chain_id] = chain_idx
            chain_idx += 1

        for res in chain.get_residues():
            # accept both ATOM (" ") and HETATM
            het_flag = res.get_id()[0]
            if het_flag == "W" or het_flag == "H_HOH":
                continue   # skip water

            res_name = res.get_resname().strip()
            res_num  = res.get_id()[1]

            # skip non-standard residues
            if res_name not in THREE_TO_ONE:
                continue

            res_coords = np.zeros((37, 3), dtype=np.float32)
            for atom in res.get_atoms():
                atom_name = atom.get_name().strip()
                if atom_name in ATOM37_NAMES:
                    idx             = ATOM37_NAMES.index(atom_name)
                    res_coords[idx] = atom.get_coord()

            coords_list.append(res_coords)
            residues_list.append(res_name)
            residue_id_list.append(f"{chain_id}:{res_num}")
            chains_list.append(chain_map[chain_id])

    if len(coords_list) == 0:
        raise ValueError(f"No valid residues found in {pdb_path}")

    data            = Data()
    data.coords     = torch.tensor(np.stack(coords_list), dtype=torch.float32)
    data.residues   = residues_list
    data.residue_id = residue_id_list
    data.chains     = torch.tensor(chains_list, dtype=torch.long)
    data.id         = os.path.splitext(os.path.basename(pdb_path))[0]

    return data


# ======================================================
# Label loading -- residue-level binary labels
# ======================================================

import sys as _sys
_sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from label_formats import detect_format, parse_triplet_file, parse_site_file

def parse_label_file(path):
    """Parse a triplet-format label file (Protein/Peptide/DNA/RNA):
    '>header / sequence / 0101...labels' triplets into
    {header: {"sequence": str, "labels": str}}.

    For Metal's site-format files, use collect_records() instead, which
    auto-detects the format and returns a compatible structure.
    """
    return parse_triplet_file(path)


def collect_records(task_dir):
    """Find every .txt / .fa label file in task_dir and parse all records,
    auto-detecting per-file whether it's triplet format (Protein/Peptide/
    DNA/RNA) or site format (Metal).

    Returns {record_id: {"sequence": str or None, "labels": str or None,
                          "positions": set(int) or None}}
    -- exactly one of "labels" (triplet) or "positions" (site) will be set,
    the other is None; downstream code (build_residue_labels) checks which.
    """
    label_files = sorted(
        glob.glob(os.path.join(task_dir, "*.txt")) +
        glob.glob(os.path.join(task_dir, "*.fa"))
    )
    print(f"Found {len(label_files)} label file(s) in {task_dir}: "
          f"{[os.path.basename(f) for f in label_files]}")

    records = {}
    for path in label_files:
        fmt = detect_format(path)

        if fmt == "triplet":
            parsed = parse_triplet_file(path)
            print(f"  {os.path.basename(path)} [triplet]: {len(parsed)} records")
            for rid, rec in parsed.items():
                records[rid] = {
                    "sequence": rec["sequence"],
                    "labels": rec["labels"],
                    "positions": None,
                    "format": "triplet",
                }
        else:
            parsed = parse_site_file(path)
            print(f"  {os.path.basename(path)} [site]: {len(parsed)} records")
            for rid, positions in parsed.items():
                if rid in records and records[rid]["positions"] is not None:
                    # merge positions if this record_id appears in multiple
                    # site-format files (e.g. train + test both list it)
                    records[rid]["positions"] |= positions
                else:
                    records[rid] = {
                        "sequence": None,
                        "labels": None,
                        "positions": positions,
                        "format": "site",
                    }

    return records


def build_residue_labels(process_data, rec, record_id):
    """Build a (N_res,) 0/1 residue label tensor for either label format.

    - triplet format (rec["labels"] is a string): positional alignment
      between the label string and process_data.residues, as before.
    - site format (rec["positions"] is a set of ints): matches each
      residue's actual PDB residue NUMBER (from process_data.residue_id,
      e.g. 'A:271' -> 271) against the positive-position set. This does
      NOT require the label count to match residue count, since positions
      are matched by number, not sequence order.

    Returns None (with a warning) if labels can't be reliably aligned.
    """
    n_res = len(process_data.residues)

    if rec["format"] == "triplet":
        labels_str = rec["labels"]
        n_lab = len(labels_str)

        if n_res != n_lab:
            print(f"[WARN] {record_id}: PDB has {n_res} residues but label "
                  f"string has {n_lab} -- skipping residue-level labels for "
                  f"this entry (structure/sequence length mismatch).")
            return None

        return torch.tensor([int(c) for c in labels_str], dtype=torch.long)

    else:  # site format
        positive_positions = rec["positions"]
        labels = []
        for res_id in process_data.residue_id:
            # res_id looks like 'A:271' -- extract the residue number
            try:
                res_num = int(res_id.split(":")[1])
            except (IndexError, ValueError):
                labels.append(0)
                continue
            labels.append(1 if res_num in positive_positions else 0)

        n_matched = sum(labels)
        if n_matched == 0:
            print(f"[WARN] {record_id}: none of {len(positive_positions)} "
                  f"positive residue position(s) matched any residue in the "
                  f"PDB structure -- check residue numbering alignment.")

        return torch.tensor(labels, dtype=torch.long)


# ======================================================
# Feature extraction
# ======================================================

def build_surface_graph(mesh, device):
    vertices   = np.asarray(mesh.vertices)
    faces      = np.asarray(mesh.faces)
    x          = torch.tensor(vertices, dtype=torch.float32)
    x          = normalize_coords(x)

    i = faces[:, 0]; j = faces[:, 1]; k = faces[:, 2]
    edges      = np.stack([
        np.concatenate([i, j, k, j, k, i]),
        np.concatenate([j, k, i, i, j, k])
    ], axis=0)
    edge_index = torch.from_numpy(edges).long()
    edge_index = torch.unique(edge_index, dim=1)

    row, col   = edge_index
    edge_attr  = torch.norm(x[row] - x[col], dim=1, keepdim=True)
    face_index = torch.from_numpy(faces.T).long()

    return (
        x.to(device),
        edge_index.to(device),
        face_index.to(device),
        edge_attr.to(device)
    )


def get_surface_features(mesh, encoder=None, device="cuda", input_dim=16):
    faces    = mesh.faces
    vertices = mesh.vertices

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)

    face_areas   = 0.5 * np.linalg.norm(cross, axis=1)
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)
    centroids    = (v0 + v1 + v2) / 3.0

    protein_center = vertices.mean(axis=0)
    dist_to_center = np.linalg.norm(centroids - protein_center, axis=1)

    # curvature
    try:
        vertex_mean = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=2.0
        )
        face_mean_curv = (
            vertex_mean[faces[:, 0]] +
            vertex_mean[faces[:, 1]] +
            vertex_mean[faces[:, 2]]
        ) / 3.0
    except Exception:
        face_mean_curv = np.zeros(len(faces))

    try:
        vertex_gauss = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=2.0
        )
        face_gauss_curv = (
            vertex_gauss[faces[:, 0]] +
            vertex_gauss[faces[:, 1]] +
            vertex_gauss[faces[:, 2]]
        ) / 3.0
    except Exception:
        face_gauss_curv = np.zeros(len(faces))

    # concavity
    outward_dir = centroids - protein_center
    outward_dir = outward_dir / (np.linalg.norm(outward_dir, axis=1, keepdims=True) + 1e-8)
    concavity   = -np.sum(face_normals * outward_dir, axis=1)

    # edge stats
    e1_len = np.linalg.norm(edge1, axis=1)
    e2_len = np.linalg.norm(edge2, axis=1)
    e3_len = np.linalg.norm(v2 - v1, axis=1)
    avg_edge_len = (e1_len + e2_len + e3_len) / 3.0
    perimeter    = e1_len + e2_len + e3_len
    aspect_ratio = 4 * np.sqrt(3) * face_areas / (perimeter ** 2 + 1e-8)

    z_coord = centroids[:, 2]

    handcrafted = np.stack([
        face_areas,
        dist_to_center,
        face_mean_curv,
        face_gauss_curv,
        concavity,
        avg_edge_len,
        aspect_ratio,
        z_coord,
    ], axis=1)

    handcrafted = (handcrafted - handcrafted.mean(0)) / (handcrafted.std(0) + 1e-6)

    if encoder is not None:
        x, edge_index, face_index, edge_attr = build_surface_graph(mesh, device)
        vertices_t     = torch.tensor(vertices, dtype=torch.float32).to(device)
        center         = vertices_t.mean(dim=0, keepdim=True)
        dist           = torch.norm(vertices_t - center, dim=1, keepdim=True)
        vertex_normals = torch.tensor(
            mesh.vertex_normals.copy(), dtype=torch.float32
        ).to(device)
        z_coord_v = vertices_t[:, 2:3]
        h         = torch.cat([vertex_normals, dist, z_coord_v], dim=1)
        h         = (h - h.mean(0)) / (h.std(0) + 1e-6)
        pad       = torch.zeros(h.size(0), input_dim - h.size(1), device=device)
        h         = torch.cat([h, pad], dim=1)
        with torch.no_grad():
            z_vertex, _ = encoder(h, x, edge_index, face_index, edge_attr)
        z_vertex = z_vertex.cpu().numpy()
        z_face   = (
            z_vertex[faces[:, 0]] +
            z_vertex[faces[:, 1]] +
            z_vertex[faces[:, 2]]
        ) / 3.0
        return np.concatenate([z_face, handcrafted], axis=1).astype(np.float32)
    else:
        return handcrafted.astype(np.float32)


def get_atom_features(process_data, encoder=None, device="cuda", k=8):
    coords_np        = process_data.coords.numpy()
    residues         = process_data.residues
    elements         = ["C", "N", "O", "S", "P", "H"]
    elem_to_idx      = {e: i for i, e in enumerate(elements)}
    backbone_indices = {0, 1, 2, 3}

    coords_list   = []
    features_list = []
    backbone_list = []
    radii_list    = []
    hydro_list    = []
    hbond_d_list  = []
    hbond_a_list  = []
    aromatic_list = []
    charge_list   = []

    for res_idx in range(coords_np.shape[0]):
        res_name = residues[res_idx] if res_idx < len(residues) else "GLY"

        for atom_idx in range(coords_np.shape[1]):
            x, y, z   = coords_np[res_idx, atom_idx]
            if np.linalg.norm([x, y, z]) < 1e-4:
                continue
            atom_name = ATOM37_NAMES[atom_idx]
            element   = atom_name[0]
            if element == "H":
                continue

            coords_list.append([x, y, z])

            # element one-hot
            one_hot = np.zeros(len(elements), dtype=np.float32)
            if element in elem_to_idx:
                one_hot[elem_to_idx[element]] = 1.0
            features_list.append(one_hot)

            backbone_list.append(float(atom_idx in backbone_indices))
            radii_list.append(ELEMENT_RADII.get(element, 1.8))

            hydro_list.append(HYDROPHOBICITY.get(res_name, 0.0))
            hbond_d_list.append(float(atom_name in HBOND_DONOR))
            hbond_a_list.append(float(atom_name in HBOND_ACCEPTOR))
            aromatic_list.append(float(res_name in AROMATIC_RES))

            if res_name in CHARGED_POS:
                charge_list.append(1.0)
            elif res_name in CHARGED_NEG:
                charge_list.append(-1.0)
            else:
                charge_list.append(0.0)

    if len(coords_list) == 0:
        raise ValueError("No valid atoms found.")

    coords      = np.array(coords_list,   dtype=np.float32)
    features    = np.array(features_list, dtype=np.float32)
    is_backbone = np.array(backbone_list, dtype=np.float32)[:, None]
    radii       = np.array(radii_list,    dtype=np.float32)[:, None]
    radii       = (radii - radii.mean()) / (radii.std() + 1e-6)

    hydro    = np.array(hydro_list,   dtype=np.float32)[:, None]
    hbond_d  = np.array(hbond_d_list, dtype=np.float32)[:, None]
    hbond_a  = np.array(hbond_a_list, dtype=np.float32)[:, None]
    aromatic = np.array(aromatic_list, dtype=np.float32)[:, None]
    charge   = np.array(charge_list,  dtype=np.float32)[:, None]

    # normalize hydrophobicity and charge
    hydro  = (hydro  - hydro.mean())  / (hydro.std()  + 1e-6)
    charge = (charge - charge.mean()) / (charge.std() + 1e-6)

    # geometric features
    tree = cKDTree(coords)

    protein_center = coords.mean(axis=0)
    dist_to_center = np.linalg.norm(coords - protein_center, axis=1, keepdims=True)
    dist_norm      = dist_to_center / (dist_to_center.max() + 1e-6)

    dists_knn, _   = tree.query(coords, k=2)
    nn_dist        = dists_knn[:, 1:].astype(np.float32)

    neighbors_5  = np.array([
        len(tree.query_ball_point(c, r=5.0)) - 1
        for c in coords
    ], dtype=np.float32)[:, None]

    neighbors_8  = np.array([
        len(tree.query_ball_point(c, r=8.0)) - 1
        for c in coords
    ], dtype=np.float32)[:, None]

    buriedness_5 = neighbors_5 / (neighbors_5.max() + 1e-6)
    buriedness_8 = neighbors_8 / (neighbors_8.max() + 1e-6)

    geo_features = np.concatenate([
        dist_norm,
        nn_dist,
        buriedness_5,
        buriedness_8,
    ], axis=1)
    geo_features = (geo_features - geo_features.mean(0)) / (geo_features.std(0) + 1e-6)

    all_features = np.concatenate([
        features,     # 6
        is_backbone,  # 1
        radii,        # 1
        hydro,        # 1
        hbond_d,      # 1
        hbond_a,      # 1
        aromatic,     # 1
        charge,       # 1
        geo_features, # 4
    ], axis=1)        # total: 17

    if encoder is not None:
        coords_t   = torch.from_numpy(coords).float().to(device)
        h          = torch.from_numpy(all_features).float().to(device)
        h_mean     = coords_t.mean(dim=0, keepdim=True)
        h_std      = coords_t.std(dim=0, keepdim=True).clamp(min=1e-6)
        coords_t   = (coords_t - h_mean) / h_std
        edge_index = knn_graph(coords_t, k=k, loop=False)
        row, col   = edge_index
        dist       = torch.norm(coords_t[row] - coords_t[col], dim=1, keepdim=True)
        with torch.no_grad():
            z, _ = encoder(h, coords_t, edge_index, dist)
        return z.cpu().numpy().astype(np.float32)
    else:
        return all_features.astype(np.float32)


def amino_acid_one_hot(residues):
    idx = [AA_TO_IDX.get(r, 0) for r in residues]
    return F.one_hot(torch.tensor(idx), num_classes=len(AA_LIST)).float()


def compute_alpha(coords):
    v1    = coords[1:-1] - coords[:-2]
    v2    = coords[2:]   - coords[1:-1]
    cos_a = F.cosine_similarity(v1, v2, dim=-1)
    angle = torch.acos(torch.clamp(cos_a, -1.0, 1.0))
    angle = F.pad(angle, (1, 1))
    return angle.unsqueeze(-1)


def compute_kappa(coords):
    v1         = coords[1:-1] - coords[:-2]
    v2         = coords[2:]   - coords[1:-1]
    cross      = torch.cross(v1, v2, dim=-1)
    norm_cross = torch.norm(cross, dim=-1)
    norm_v1    = torch.norm(v1, dim=-1) + 1e-6
    norm_v2    = torch.norm(v2, dim=-1) + 1e-6
    kappa      = norm_cross / (norm_v1 * norm_v2)
    kappa      = F.pad(kappa, (1, 1))
    return kappa.unsqueeze(-1)


def compute_dihedrals(coords):
    p0 = coords[:-3]; p1 = coords[1:-2]
    p2 = coords[2:-1]; p3 = coords[3:]
    b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
    b1 = F.normalize(b1, dim=-1)
    v  = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w  = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x  = (v * w).sum(-1)
    y  = torch.cross(b1, v, dim=-1).mul(w).sum(-1)
    angle = torch.atan2(y, x)
    angle = F.pad(angle, (1, 2))
    return angle.unsqueeze(-1)


def build_residue_data(process_data):
    coords_np = process_data.coords.numpy()
    ca_coords = torch.tensor(coords_np[:, 1, :], dtype=torch.float32)
    data          = Data()
    data.coords   = ca_coords
    data.residues = list(process_data.residues)
    data.batch    = torch.zeros(ca_coords.size(0), dtype=torch.long)
    return data


def get_residue_features(process_data, pdb_path=None, esm2_model=None,
                          batch_converter=None, device="cuda",
                          use_dssp=True, use_esm=True):
    """Per-residue feature vector: aa one-hot (20) + alpha/kappa/dihedral
    (3) + physicochemical (10) + phi/psi/RSA (5, optional) + per-residue
    ESM-2 (1280, optional) = ~1318 total.
    """
    res_data = build_residue_data(process_data)
    n_res    = len(res_data.residues)

    feats = []
    for feature in ["amino_acid_one_hot", "alpha", "kappa", "dihedrals"]:
        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(res_data.residues))
        elif feature == "alpha":
            feats.append(compute_alpha(res_data.coords))
        elif feature == "kappa":
            feats.append(compute_kappa(res_data.coords))
        elif feature == "dihedrals":
            feats.append(compute_dihedrals(res_data.coords))
    feats = [f if f.ndim == 2 else f.unsqueeze(-1) for f in feats]

    # physicochemical
    physchem = torch.tensor(get_physchem_features(res_data.residues), dtype=torch.float32)
    feats.append(physchem)

    # phi/psi + RSA
    if use_dssp and pdb_path is not None:
        structural_feats = get_residue_structural_features(process_data, pdb_path)
        feats.append(torch.tensor(structural_feats, dtype=torch.float32))

    # per-residue ESM-2 embedding
    if use_esm and esm2_model is not None and batch_converter is not None:
        sequence = get_sequence(res_data.residues)
        esm_feats = get_per_residue_esm_features(
            sequence, esm2_model, batch_converter, device=device
        )
        if len(esm_feats) != n_res:
            print(f"[WARN] per-residue ESM output length {len(esm_feats)} != "
                  f"{n_res} residues -- truncating/padding.")
            if len(esm_feats) < n_res:
                pad = np.zeros((n_res - len(esm_feats), esm_feats.shape[1]), dtype=np.float32)
                esm_feats = np.concatenate([esm_feats, pad], axis=0)
            else:
                esm_feats = esm_feats[:n_res]
        feats.append(torch.tensor(esm_feats, dtype=torch.float32))

    return torch.cat(feats, dim=-1)


def get_sse_features(partition, sse_labels):
    N_sse    = partition.shape[1]
    Pi_dense = partition.toarray()
    sse_labels = np.array(sse_labels)
    one_hot  = np.zeros((N_sse, 3),  dtype=np.float32)
    lengths  = np.zeros((N_sse, 1),  dtype=np.float32)
    for i in range(N_sse):
        idx = Pi_dense[:, i] > 0
        if idx.sum() == 0:
            continue
        one_hot[i, sse_labels[idx][0]] = 1.0
        lengths[i, 0] = idx.sum()
    lengths = (lengths - lengths.mean()) / (lengths.std() + 1e-6)
    return np.concatenate([one_hot, lengths], axis=1)


def get_protein_features(protein_seq: str, device: str = "cuda") -> np.ndarray:
    data     = [("protein", protein_seq)]
    _, _, tokens = batch_converter(data)
    tokens   = tokens.to(device)
    with torch.no_grad():
        results = esm2_model(tokens, repr_layers=[33], return_contacts=False)
    cls_embedding = results["representations"][33][:, 0, :]
    return cls_embedding.cpu().numpy()


def get_sequence(residues):
    return "".join([THREE_TO_ONE.get(r.upper(), "X") for r in residues])


# ======================================================
# Graph data structures
# ======================================================

@dataclass
class ProteinGraphLevel:
    name: str
    A:    object
    X:    torch.Tensor

    def to_torch(self, device):
        if isinstance(self.A, list):
            if isinstance(self.A[0], torch.Tensor):
                self.A = [a.to(device) for a in self.A]
            else:
                converted = []
                for A_i in self.A:
                    A_coo   = A_i.tocoo()
                    indices = torch.stack([
                        torch.from_numpy(A_coo.row),
                        torch.from_numpy(A_coo.col)
                    ]).long()
                    values  = torch.from_numpy(A_coo.data).float()
                    converted.append(
                        torch.sparse_coo_tensor(
                            indices, values, size=A_coo.shape, device=device
                        ).coalesce()
                    )
                self.A = converted
        else:
            if isinstance(self.A, torch.Tensor):
                self.A = self.A.to(device)
            else:
                A       = self.A.tocoo()
                indices = torch.stack([
                    torch.from_numpy(A.row),
                    torch.from_numpy(A.col)
                ]).long()
                values  = torch.from_numpy(A.data).float()
                self.A  = torch.sparse_coo_tensor(
                    indices, values, size=A.shape, device=device
                ).coalesce()

        if self.X is not None:
            self.X = self.X.to(device)

    @property
    def num_nodes(self):
        A = self.A[0] if isinstance(self.A, list) else self.A
        return A.shape[0]


class HierarchicalProteinGraph:
    def __init__(
        self,
        surface:           ProteinGraphLevel,
        atom:              ProteinGraphLevel,
        residue:           ProteinGraphLevel,
        sse:               ProteinGraphLevel,
        protein:           ProteinGraphLevel,
        partitions:        dict,
        residue_labels:    torch.Tensor = None,
        residue_id:        list        = None,
    ):
        self.surface        = surface
        self.atom            = atom
        self.residue         = residue
        self.sse             = sse
        self.protein         = protein
        self.partitions      = partitions
        self.residue_labels  = residue_labels
        self.residue_id      = residue_id
        self._validate()

    def to_torch(self, device):
        self.surface.to_torch(device)
        self.atom.to_torch(device)
        self.residue.to_torch(device)
        self.sse.to_torch(device)
        self.protein.to_torch(device)

        new_partitions = {}
        for key, Pi in self.partitions.items():
            if key.endswith("_T"):
                continue
            if isinstance(Pi, torch.Tensor):
                Pi_t = Pi.to(device)
            else:
                Pi      = Pi.tocoo()
                indices = torch.stack([
                    torch.from_numpy(Pi.row),
                    torch.from_numpy(Pi.col)
                ]).long()
                values  = torch.from_numpy(Pi.data).float()
                Pi_t    = torch.sparse_coo_tensor(
                    indices, values, size=Pi.shape, device=device
                ).coalesce()
            new_partitions[key]        = Pi_t
            new_partitions[key + "_T"] = Pi_t.transpose(0, 1).coalesce()
        self.partitions = new_partitions

        if self.residue_labels is not None:
            self.residue_labels = self.residue_labels.to(device)

    def _validate(self):
        required = {"surface_to_atom", "atom_to_residue", "residue_to_sse"}
        missing  = required - self.partitions.keys()
        if missing:
            raise ValueError(f"Missing partitions: {missing}")
        Pi_sa = self.partitions["surface_to_atom"]
        Pi_ar = self.partitions["atom_to_residue"]
        Pi_rs = self.partitions["residue_to_sse"]
        assert self.surface.num_nodes == Pi_sa.shape[0]
        assert self.atom.num_nodes    == Pi_sa.shape[1]
        assert self.atom.num_nodes    == Pi_ar.shape[0]
        assert self.residue.num_nodes == Pi_ar.shape[1]
        assert self.residue.num_nodes == Pi_rs.shape[0]
        assert self.sse.num_nodes     == Pi_rs.shape[1]
        if self.residue_labels is not None:
            assert self.residue.num_nodes == self.residue_labels.shape[0], (
                f"residue_labels length {self.residue_labels.shape[0]} != "
                f"residue node count {self.residue.num_nodes}"
            )


def build_hierarchical_protein_graph(
        surface_path:    str,
        process_data,
        pdb_path:        str           = None,
        residue_labels:  torch.Tensor = None,
        surface_k:       int          = 8,
        atom_encoder                  = None,
        surface_encoder               = None,
        device:          str          = "cuda"
) -> HierarchicalProteinGraph:

    partitions, sse_labels = extract_partition_matrices(
        surface_path=surface_path,
        process_data=process_data,
    )

    mesh = trimesh.load(surface_path, process=False)
    mesh = mesh_simplification_quadric_decimation(
        mesh, target_faces=partitions['surface_to_atom'].shape[0]
    )
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)

    # Surface
    A_surface     = build_surface_knn_adjacency(face_centroids, k=surface_k)
    X_surface     = get_surface_features(mesh, encoder=surface_encoder, device=device)
    X_surface     = torch.tensor(X_surface, dtype=torch.float32)
    surface_level = ProteinGraphLevel("surface", A_surface, X_surface)

    # Atom
    Pi_sa      = partitions["surface_to_atom"]
    A_atom_raw = coarsen_adjacency(A_surface, Pi_sa)
    A_atom_knn = build_knn_adjacency(get_atom_coords(process_data), k=8)
    X_atom     = get_atom_features(process_data, encoder=atom_encoder, device=device)
    X_atom     = torch.tensor(X_atom, dtype=torch.float32)
    atom_level = ProteinGraphLevel(
        "atom",
        A=[normalize_adjacency(A_atom_raw), normalize_adjacency(A_atom_knn)],
        X=X_atom
    )

    # Residue
    Pi_ar     = partitions["atom_to_residue"]
    A_res_raw = coarsen_adjacency(A_atom_raw, Pi_ar)
    A_res_knn = build_knn_adjacency(process_data.coords.numpy()[:, 1, :], k=10)
    X_res     = get_residue_features(
        process_data, pdb_path=pdb_path,
        esm2_model=esm2_model, batch_converter=batch_converter, device=device
    )
    residue_level = ProteinGraphLevel(
        "residue",
        A=[normalize_adjacency(A_res_raw), normalize_adjacency(A_res_knn)],
        X=X_res
    )

    # SSE
    Pi_rs     = partitions["residue_to_sse"]
    A_sse_raw = coarsen_adjacency(A_res_raw, Pi_rs)
    A_sse     = normalize_adjacency(A_sse_raw)
    X_sse     = torch.tensor(get_sse_features(Pi_rs, sse_labels), dtype=torch.float32)
    sse_level = ProteinGraphLevel("sse", A_sse, X_sse)

    # Protein
    Pi_sp         = partitions["sse_to_protein"]
    A_protein_raw = coarsen_adjacency(A_sse_raw, Pi_sp)
    A_protein     = normalize_adjacency(A_protein_raw)
    protein_seq   = get_sequence(process_data.residues)
    X_protein_np  = get_protein_features(protein_seq, device=device)
    X_protein     = torch.tensor(X_protein_np, dtype=torch.float32)
    protein_level = ProteinGraphLevel("protein", A_protein, X=X_protein)

    return HierarchicalProteinGraph(
        surface=surface_level,
        atom=atom_level,
        residue=residue_level,
        sse=sse_level,
        protein=protein_level,
        partitions=partitions,
        residue_labels=residue_labels,
        residue_id=list(process_data.residue_id),
    )


# ======================================================
# Main -- Protein / Peptide / DNA / RNA / Metal
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hierarchical graph construction for Protein/Peptide/DNA/RNA/Metal tasks"
    )

    parser.add_argument("--task_dir", type=str, required=True,
                         help="Task dir containing label .txt/.fa files and a PDBs/ subfolder")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--atom_encoder_path", type=str, default=None)
    parser.add_argument("--surface_encoder_path", type=str, default=None)

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load encoders
    # --------------------------------------------------
    atom_encoder    = None
    surface_encoder = None

    if args.atom_encoder_path:
        print(f"Loading atom encoder from {args.atom_encoder_path}")
        ckpt         = torch.load(args.atom_encoder_path, map_location="cpu")
        atom_cfg     = ckpt["config"]["atom"]
        atom_encoder = EGNN(
            in_node_nf=atom_cfg["atom_feat_dim"],
            hidden_nf=atom_cfg["hidden_dim"],
            out_node_nf=atom_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=atom_cfg["n_layers"]
        )
        atom_encoder.load_state_dict(ckpt["model_state_dict"])
        atom_encoder = atom_encoder.to(device).eval()

    if args.surface_encoder_path:
        print(f"Loading surface encoder from {args.surface_encoder_path}")
        ckpt            = torch.load(args.surface_encoder_path, map_location="cpu")
        surface_cfg     = ckpt["config"]["surface"]
        surface_encoder = EMNN(
            in_node_nf=surface_cfg["input_dim"],
            hidden_nf=surface_cfg["hidden_dim"],
            out_node_nf=surface_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=surface_cfg["n_layers"]
        )
        surface_encoder.load_state_dict(ckpt["model_state_dict"])
        surface_encoder = surface_encoder.to(device).eval()

    # --------------------------------------------------
    # Load records (sequence + labels) and locate PDBs
    # --------------------------------------------------
    records = collect_records(args.task_dir)
    print(f"Total records: {len(records)}")

    # PDBs live in different places depending on format: triplet-format
    # tasks (Protein/Peptide/DNA/RNA) use PDBs/{record_id}.pdb (from
    # folding.py); site-format tasks (Metal) ship real structures under
    # PDBs/receptor/{record_id}.pdb.
    pdb_dir_flat     = os.path.join(args.task_dir, "PDBs")
    pdb_dir_receptor = os.path.join(args.task_dir, "PDBs", "receptor")

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Process
    # --------------------------------------------------
    errors = []

    for record_id, rec in tqdm(records.items(), desc=f"Processing {os.path.basename(args.task_dir)}"):

        output_path = os.path.join(args.output_dir, f"{record_id}.pt")
        if os.path.exists(output_path):
            continue

        if rec["format"] == "site":
            pdb_path = os.path.join(pdb_dir_receptor, f"{record_id}.pdb")
        else:
            pdb_path = os.path.join(pdb_dir_flat, f"{record_id}.pdb")
        surface_path = os.path.join(args.output_dir, f"_tmp_{record_id}.obj")

        if not os.path.exists(pdb_path):
            print(f"Missing PDB: {record_id}")
            errors.append((record_id, "missing PDB"))
            continue

        try:
            # ── load protein ──────────────────────────
            process_data = load_protein_from_pdb(pdb_path)

            # ── export surface ────────────────────────
            export_surface(
                pdb_path=pdb_path,
                output_path=surface_path,
                surface_quality=0,
                solvent_radius=1.4,
                selection="all"
            )

            # ── residue-level labels ──────────────────
            if rec["format"] == "site":
                residue_labels = None
            else:
                residue_labels = build_residue_labels(
                    process_data, rec, record_id
                )

            # ── build graph ───────────────────────────
            graph = build_hierarchical_protein_graph(
                surface_path=surface_path,
                process_data=process_data,
                pdb_path=pdb_path,
                residue_labels=residue_labels,
                atom_encoder=atom_encoder,
                surface_encoder=surface_encoder,
                device=device
            )

            torch.save(graph, output_path)

        except Exception as e:
            print(f"Error {record_id}: {e}")
            errors.append((record_id, str(e)))

        finally:
            if os.path.exists(surface_path):
                os.remove(surface_path)

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        for rid, err in errors[:10]:
            print(f"  {rid}: {err}")