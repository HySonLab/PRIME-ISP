"""
extract_case_study_labels.py

Case Studies section (thrombin, ERalpha, calmodulin): single-structure design.

Rather than folding each target from sequence and stitching together
interface labels from several DIFFERENT partner-complex PDBs (which needed
a folded-structure numbering <-> crystal-structure numbering alignment step
for every single partner -- a real source of subtle bugs), each target here
uses ONE real co-crystal structure that happens to have MULTIPLE, spatially
distinct partner classes bound to it AT THE SAME TIME:

    thrombin  : PDB 8TQS  (thrombin H+L, bound simultaneously to a DNA
                aptamer at exosite II AND a dabigatran-derived small
                molecule at the active site -- delivered as one bivalent
                conjugate, but the two ends occupy two distinct sites)
    er_alpha  : PDB 3L03  (ERalpha LBD, bound simultaneously to estetrol in
                the orthosteric pocket AND a GRIP1 coactivator peptide at
                the AF-2 groove)
    calmodulin: PDB 1CDL  (calmodulin, ONE continuous 147-residue chain,
                bound simultaneously to 4 Ca2+ ions at the EF-hand loops
                AND a CaMKII-alpha-derived target peptide at the
                hydrophobic groove)

(An earlier version of this used HIV-1 RT / 4Q0B instead of calmodulin.
Dropped: RT's two chains (p66+p51, ~1000 residues combined) get
concatenated into one un-delimited string before ESM-2 embedding --
see utils/hierarchical_graph.py's get_sequence/load_protein_from_pdb,
which has no separator between chains -- a plausible confound for a
target this large, and its RNA/NNRTI sites came back near-zero signal
at every threshold tested. Calmodulin is a single chain, sidestepping
that issue rather than requiring a fix to shared graph-building code.)

For each target we:
  1. Download the one representative complex.
  2. For each site, find target-chain residues within `threshold` Angstrom
     of that site's partner atoms (a chain, or a named HETATM ligand).
  3. Write a CLEAN protein-only PDB (only the target chain(s); every
     partner -- DNA/RNA/peptide/ligand/waters -- stripped out) as the
     actual PRIME-ISP input. Numbering is untouched, so it is exactly the
     structure's own auth_seq_id numbering -- no folding, no cross-file
     sequence alignment, no numbering-translation step needed anywhere in
     this pipeline.
  4. Write one JSON label file per site, positions given directly in that
     same auth_seq_id numbering.

ASYMMETRIC-UNIT DUPLICATE HANDLING: 3L03 and 1CDL each contain multiple
copies of the whole complex (1CDL has 4: A/B, C/D, E/F, G/H). A partner
spec's "chain" value can be a single chain id
or a list of CANDIDATE chain ids; the script tries them against the chosen
target_chains in order and uses the first candidate that actually yields a
non-empty, non-trivial contact set, printing which one it picked. This
means a wrong guess about which lettered copy pairs with which fails loudly
(or silently self-corrects) instead of quietly returning an empty/wrong
label set for the wrong asymmetric-unit copy.

PLEASE STILL SANITY-CHECK the printed interface-residue counts per site
against the structure once (e.g. in PyMOL) before treating these labels as
final -- the chain letters and ligand codes below were verified against
each entry's raw PDB header (COMPND/HETATM records), which is reliable,
but "some residues near a chain" and "the biologically correct interface"
aren't automatically identical (e.g. crystal-packing contacts).

Requires: biopython, network access to files.rcsb.org.

Usage:
    python utils/extract_case_study_labels.py \
        --output_dir ./Datasets/CaseStudies/labels \
        --pdb_dir ./Datasets/CaseStudies/clean_pdbs \
        --pdb_cache_dir ./Datasets/CaseStudies/raw_complexes
"""

import argparse
import json
import os
import urllib.request

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

# ======================================================
# Case-study definitions -- verified against 8TQS / 3L03 / 4Q0B raw PDB headers.
# ======================================================
# partner: ("chain", chain_id_or_list_of_candidates) for a peptide/DNA/RNA
#          partner, or ("resname", 3-letter_code) for a small-molecule ligand.
CASE_STUDIES = {
    "thrombin": {
        "pdb_id": "8TQS",
        "target_chains": ["H", "L"],  # heavy + light chain of mature thrombin
        "sites": {
            "DPI_exosite_II": {
                "partner": ("chain", "D"),
                "note": "DNA aptamer HD22 (exosite II half of the bivalent "
                        "HD22-dabigatran conjugate).",
            },
            "SMPI_active_site": {
                "partner": ("resname", "4CC"),
                "note": "Dabigatran-derived small-molecule inhibitor "
                        "(active-site half of the same bivalent conjugate).",
            },
        },
    },
    "er_alpha": {
        "pdb_id": "3L03",
        "target_chains": ["A"],  # use one copy of the LBD only
        "sites": {
            "SMPI_orthosteric": {
                "partner": ("resname", "4OH"),  # estetrol
                "note": "Estetrol bound in the orthosteric ligand-binding pocket.",
            },
            "PepPI_coactivator": {
                "partner": ("chain", ["C", "D"]),  # GRIP1 peptide, ASU copy TBD
                "note": "GRIP1 NR-box II coactivator peptide bound at the AF-2 groove.",
            },
        },
    },
    "calmodulin": {
        "pdb_id": "1CDL",
        "target_chains": ["A"],  # one continuous 147-residue calmodulin chain
        "sites": {
            "MPI_calcium": {
                "partner": ("resname", "CA"),
                "note": "4 Ca2+ ions bound in the EF-hand loops.",
            },
            "PepPI_target": {
                "partner": ("chain", ["E", "F", "G", "H"]),  # target peptide, ASU copy TBD
                "note": "CaMKII-alpha-derived target peptide bound at the "
                        "hydrophobic groove (Meador et al., Science 1992).",
            },
        },
    },
}

CONTACT_CUTOFFS = {
    "SMPI": 6.5,
    "PepPI": 5.0,
    "DPI": 5.0,
    "RPI": 5.0,
    "MPI": 3.0,
    "CarbPI": 4.2,
}
DEFAULT_CUTOFF = 4.5
MIN_INTERFACE_RESIDUES = 3


def get_cutoff(site_name, override=None):
    if override is not None:
        return override
    prefix = site_name.split("_")[0]
    if prefix not in CONTACT_CUTOFFS:
        print(f"    [note] site '{site_name}' has no CONTACT_CUTOFFS entry for "
              f"prefix '{prefix}' -- using default {DEFAULT_CUTOFF} A.")
    return CONTACT_CUTOFFS.get(prefix, DEFAULT_CUTOFF)


# ======================================================
# PDB download
# ======================================================

def download_pdb(pdb_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {pdb_id} ...")
    urllib.request.urlretrieve(url, out_path)
    return out_path


# ======================================================
# Interface residues
# ======================================================

def get_partner_atoms(structure, kind, value):
    """value is a single chain id / resname, never a list -- callers resolve
    candidate lists to one concrete value first (see resolve_partner)."""
    atoms = []
    for chain in structure[0]:
        for residue in chain:
            if kind == "chain" and chain.id == value:
                atoms.extend(residue.get_atoms())
            elif kind == "resname" and residue.get_resname().strip() == value:
                atoms.extend(residue.get_atoms())
    return atoms


def interface_resnums_for_atoms(structure, target_chain_ids, partner_atoms, threshold):
    partner_coords = [a.get_coord() for a in partner_atoms]
    interface = set()  # {(chain_id, auth_seq_id), ...}
    for chain in structure[0]:
        if chain.id not in target_chain_ids:
            continue
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            for atom in residue:
                ac = atom.get_coord()
                for pc in partner_coords:
                    d2 = (ac[0]-pc[0])**2 + (ac[1]-pc[1])**2 + (ac[2]-pc[2])**2
                    if d2 <= threshold ** 2:
                        interface.add((chain.id, residue.id[1]))
                        break
                else:
                    continue
                break
    return interface


def resolve_partner(structure, target_chain_ids, partner_spec, threshold):
    """Resolves a partner spec (possibly a list of candidate chain ids) to
    the concrete choice that actually contacts target_chain_ids, and
    returns (interface_resnums, description_of_what_was_used)."""
    kind, value = partner_spec
    candidates = value if isinstance(value, list) else [value]

    best = None  # (n_residues, candidate, interface_set)
    for cand in candidates:
        atoms = get_partner_atoms(structure, kind, cand)
        if not atoms:
            print(f"    [note] no atoms found at all for {kind}={cand!r} in this structure")
            continue
        interface = interface_resnums_for_atoms(structure, target_chain_ids, atoms, threshold)
        n = len(interface)
        if len(candidates) > 1:
            print(f"    [candidate] {kind}={cand!r}: {n} interface residue(s)")
        if best is None or n > best[0]:
            best = (n, cand, interface)

    if best is None:
        raise ValueError(f"Partner spec {partner_spec} matched no atoms at all in this structure.")

    n, chosen, interface = best
    if n < MIN_INTERFACE_RESIDUES:
        print(f"    [warn] best candidate {kind}={chosen!r} only gave {n} interface "
              f"residue(s) (< {MIN_INTERFACE_RESIDUES}) -- likely the wrong "
              f"asymmetric-unit copy or a stale chain letter/ligand code; verify by eye.")
    elif len(candidates) > 1:
        print(f"    [picked] {kind}={chosen!r} ({n} interface residues)")

    description = f"{kind} {chosen}" if kind == "chain" else f"ligand {chosen}"
    return interface, description


# ======================================================
# Clean (partner-stripped) PDB output
# ======================================================

class KeepChains(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_residue(self, residue):
        return is_aa(residue, standard=True)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def write_clean_pdb(structure, target_chain_ids, out_path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, KeepChains(target_chain_ids))


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Where to write per-site label JSON files.")
    parser.add_argument("--pdb_dir", type=str, required=True,
                         help="Where to write the clean, partner-stripped "
                              "PDB files to use as PRIME-ISP input.")
    parser.add_argument("--pdb_cache_dir", type=str, required=True,
                         help="Where to cache the downloaded full complexes.")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override CONTACT_CUTOFFS with ONE fixed cutoff for "
                              "every site (for sensitivity testing). Omit to use "
                              "the per-interaction-class defaults in CONTACT_CUTOFFS.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)

    parser_pdb = PDBParser(QUIET=True)

    for target_name, spec in CASE_STUDIES.items():
        print(f"\n=== {target_name} ({spec['pdb_id']}) ===")
        pdb_path = download_pdb(spec["pdb_id"], args.pdb_cache_dir)
        structure = parser_pdb.get_structure(target_name, pdb_path)
        target_chain_ids = spec["target_chains"]

        clean_path = os.path.join(args.pdb_dir, f"{target_name}_clean.pdb")
        write_clean_pdb(structure, target_chain_ids, clean_path)
        print(f"  Clean input structure: {clean_path} (chains {target_chain_ids}, "
              f"all partners/ligands/waters stripped)")

        result = {}
        for site_name, site_spec in spec["sites"].items():
            cutoff = get_cutoff(site_name, args.threshold)
            print(f"  {site_name} (cutoff={cutoff} A): {site_spec['note']}")
            interface, used = resolve_partner(
                structure, target_chain_ids, site_spec["partner"], cutoff
            )
            by_chain = {}
            for chain_id, resnum in sorted(interface):
                by_chain.setdefault(chain_id, []).append(resnum)
            for chain_id in by_chain:
                by_chain[chain_id].sort()

            result[site_name] = {
                "source": f"{spec['pdb_id']} target chains {target_chain_ids} vs. {used}",
                "note": site_spec["note"],
                "cutoff_angstrom": cutoff,
                "n_residues": len(interface),
                "positions_by_chain": by_chain,
            }

        out_path = os.path.join(args.output_dir, f"{target_name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    print("\nDone. Feed each *_clean.pdb into PRIME-ISP as-is (no folding needed); "
          "score predictions against the matching site's positions_by_chain.")


if __name__ == "__main__":
    main()