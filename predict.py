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
    "protein":        "PPI-site_Train9982-Test355",
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
    from utils.folding import load_model, try_fold, free_gpu_memory

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
        for partner, p in probs[name].items():
            write_bfactor_pdb(structures[name],
                              os.path.join(d, f"{name}_{partner}.pdb"),
                              graph.residue_id, p)
            n_sites = int((p >= args.threshold).sum())
            print(f"  {name} / {partner}: {n_sites} of {len(p)} residues "
                  f"predicted as interacting (p >= {args.threshold})")
        print(f"  results: {d}")


if __name__ == "__main__":
    main()
