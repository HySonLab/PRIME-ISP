import os
import re

def detect_format(path):
    """Return 'triplet' or 'site' based on the file's first non-blank line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return "triplet" if line.startswith(">") else "site"
    return "triplet"  # empty file, doesn't matter

# ======================================================
# Triplet format
# ======================================================

def parse_triplet_file(path):
    """Returns {record_id: {"sequence": str, "labels": str}}."""
    records = {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip() != ""]

    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.startswith(">"):
            i += 1
            continue
        record_id = header[1:].strip()
        if i + 2 < len(lines):
            sequence = lines[i + 1].strip()
            labels   = lines[i + 2].strip()
            records[record_id] = {"sequence": sequence, "labels": labels}
        i += 3

    return records


def get_ids_triplet(path):
    return list(parse_triplet_file(path).keys())


# ======================================================
# Site format (Metal)
# ======================================================

def parse_site_file(path):
    entries = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            header, rest = line.split("\t", 1)
            parts = header.split(":")
            if len(parts) < 2:
                continue
            pdb_id, chain = parts[0], parts[1]
            record_id = f"{pdb_id}{chain}"

            positions = set()
            for group in rest.split(":"):
                tokens = group.strip().split()
                if len(tokens) <= 1:
                    continue
                for res_token in tokens[1:]:  # skip the 'A_A' chain-pair token
                    m = re.match(r"^[A-Za-z]+(-?\d+)$", res_token)
                    if m:
                        positions.add(int(m.group(1)))

            entries.setdefault(record_id, set()).update(positions)

    return entries


def get_ids_site(path):
    return list(parse_site_file(path).keys())


# ======================================================
# Unified interface
# ======================================================

def get_ids(path):
    """Auto-detect format and return the list of record ids in a label file."""
    fmt = detect_format(path)
    if fmt == "triplet":
        return get_ids_triplet(path)
    else:
        return get_ids_site(path)


def load_ion_positions(label_files):
    positions = {}
    for path in label_files:
        parsed = parse_site_file(path)
        for rid, pos_set in parsed.items():
            positions.setdefault(rid, set()).update(pos_set)
    return positions