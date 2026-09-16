from addict import Dict
import numpy as np
from Bio.PDB import PDBParser
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
from Bio.PDB import Structure
import pydssp
import torch
import subprocess
from pathlib import Path
import trimesh
from Bio.PDB import Selection
import open3d as o3d

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

def mesh_simplification_quadric_decimation(mesh, target_faces):
    """
    Simplify mesh using Open3D's quadric decimation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh
    target_faces : int
        Desired number of faces after simplification

    Returns
    -------
    simplified_mesh : trimesh.Trimesh
        Simplified mesh
    """

    # Convert to Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    mesh_o3d.compute_vertex_normals()

    # Simplify using quadric decimation
    simplified_o3d = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )

    # Convert back to Trimesh
    simplified_mesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_o3d.vertices),
        faces=np.asarray(simplified_o3d.triangles),
        process=False,
    )

    return simplified_mesh

def print_structure_info(
    structure: Structure,
    include_hetero: bool = False):
    """
    Print basic information about the PDB structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    include_hetero : bool, optional
        Whether to include hetero residues (HETATM).
    """

    num_atoms = 0
    num_residues = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, _, _ = residue.id

                if not include_hetero and hetero_flag.strip():
                    continue

                num_residues += 1
                num_atoms += len(residue)

    # print("Number of residues:", num_residues)
    # print("Number of atoms:   ", num_atoms)

def build_atom_to_residue_assignment(process_data):
    """
    Build atom -> residue assignment from processed .pt data.
    """
    coords_np        = process_data.coords.numpy()  # (N_res, 37, 3)

    atom_to_residue = []

    for res_idx in range(coords_np.shape[0]):
        for atom_idx in range(coords_np.shape[1]):
            x, y, z = coords_np[res_idx, atom_idx]

            if np.linalg.norm([x, y, z]) < 1e-4:
                continue

            atom_name = ATOM37_NAMES[atom_idx]
            if atom_name[0] == "H":
                continue

            atom_to_residue.append(res_idx)

    N_atoms = len(atom_to_residue)
    N_res   = coords_np.shape[0]

    rows = np.arange(N_atoms)
    cols = np.array(atom_to_residue, dtype=np.int32)
    data = np.ones(N_atoms, dtype=np.int8)

    Pi = csr_matrix((data, (rows, cols)), shape=(N_atoms, N_res))

    assert np.all(Pi.sum(axis=1).A1 == 1)
    assert np.all(Pi.sum(axis=0).A1 > 0)

    return Pi

def build_residue_to_sse_assignment(process_data):
    """
    Build residue → SSE assignment from processed .pt data.
    Guaranteed to preserve residue count and indexing.
    """

    coords_np = process_data.coords.numpy()  # (N_res, 37, 3)
    backbone  = coords_np[:, :4, :]          # (N_res, 4, 3)
    N_res     = backbone.shape[0]

    # --------------------------------------------------
    # Determine which residues have valid backbone
    # --------------------------------------------------
    valid = np.array([
        all(np.linalg.norm(backbone[i, j]) > 1e-4 for j in range(4))
        for i in range(N_res)
    ])

    # --------------------------------------------------
    # Run DSSP only on valid residues
    # --------------------------------------------------
    labels = np.zeros(N_res, dtype=np.int32)  # default = loop (0)

    if valid.any():
        coord_tensor = torch.from_numpy(
            backbone[valid].astype(np.float32)
        )
        dssp_labels = (
            pydssp.assign(coord_tensor, out_type="index")
            .cpu()
            .numpy()
        )
        labels[valid] = dssp_labels

    # --------------------------------------------------
    # Build SSE segments (over ALL residues)
    # --------------------------------------------------
    sse_ids    = np.zeros(N_res, dtype=np.int32)
    current    = 0
    sse_ids[0] = 0

    for i in range(1, N_res):
        if labels[i] != labels[i - 1]:
            current += 1
        sse_ids[i] = current

    N_sse = current + 1

    Pi = csr_matrix(
        (
            np.ones(N_res, dtype=np.int8),
            (np.arange(N_res), sse_ids)
        ),
        shape=(N_res, N_sse)
    )

    return Pi, labels.tolist()

def get_atom_coords(process_data):
    """
    Extract valid heavy atom coordinates from processed .pt data.
    Filters out padding atoms and hydrogen.
    Used for both partition building and adjacency augmentation.
    
    Returns:
        atom_coords: np.ndarray (N_atoms, 3)
    """
    coords_np   = process_data.coords.numpy()  # (N_res, 37, 3)
    atom_coords = []

    for res_idx in range(coords_np.shape[0]):
        for atom_idx in range(coords_np.shape[1]):
            x, y, z = coords_np[res_idx, atom_idx]
            if np.linalg.norm([x, y, z]) < 1e-4:
                continue
            atom_name = ATOM37_NAMES[atom_idx]
            if atom_name[0] == "H":
                continue
            atom_coords.append([x, y, z])

    return np.array(atom_coords, dtype=np.float32)

def build_surface_to_atom_assignment(process_data, surface_path):
    """
    Build surface face -> atom assignment using atom coords from .pt data.
    """
    
    atom_coords = get_atom_coords(process_data)

    mesh           = trimesh.load(surface_path)
    mesh           = mesh_simplification_quadric_decimation(mesh, target_faces=MAX_FACES)
    face_centroids = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)

    kdtree = cKDTree(atom_coords)
    _, nearest_atom_indices = kdtree.query(face_centroids, k=1)

    N_faces = face_centroids.shape[0]
    N_atoms = atom_coords.shape[0]

    Pi = csr_matrix(
        (np.ones(N_faces, dtype=np.int8), (np.arange(N_faces), nearest_atom_indices)),
        shape=(N_faces, N_atoms)
    )

    return Pi

def extract_partition_matrices(
    surface_path: str,
    process_data,):
    partitions = {}

    Pi_surface_to_atom = build_surface_to_atom_assignment(process_data, surface_path)
    partitions["surface_to_atom"] = Pi_surface_to_atom

    Pi_atom_to_res = build_atom_to_residue_assignment(process_data)
    partitions["atom_to_residue"] = Pi_atom_to_res

    Pi_res_to_sse, sse_label = build_residue_to_sse_assignment(process_data)
    partitions["residue_to_sse"] = Pi_res_to_sse

    N_sse = Pi_res_to_sse.shape[1]
    Pi_sse_to_prot = csr_matrix(
        (
            np.ones(N_sse, dtype=np.int8),
            (np.arange(N_sse), np.zeros(N_sse, dtype=np.int32)),
        ),
        shape=(N_sse, 1),
    )
    partitions["sse_to_protein"] = Pi_sse_to_prot

    return partitions, sse_label