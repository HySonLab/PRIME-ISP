import os
import yaml
import torch
from torch.utils.data import Dataset
import torch
import yaml
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.hierarchical_graph import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

def get_sequence(residues):
    sequence = []

    for res in residues:
        res = res.upper()
        if res in THREE_TO_ONE:
            sequence.append(THREE_TO_ONE[res])
        else:
            sequence.append("X")

    sequence = "".join(sequence)

    return sequence


def _labels_from_positions(residue_id, positions):
    """Given a graph's residue_id list (e.g. ['A:271', 'A:272', ...]) and a
    set of positive residue NUMBERS (e.g. {271, 320, 322}), build a (N_res,)
    0/1 label tensor by matching residue number, not sequence position.
    Used for site-format tasks (Metal) where the graph itself is shared
    across ions and only the label differs."""
    labels = []
    for rid in residue_id:
        try:
            res_num = int(rid.split(":")[1])
        except (IndexError, ValueError):
            labels.append(0)
            continue
        labels.append(1 if res_num in positions else 0)
    return torch.tensor(labels, dtype=torch.long)


class HierarchicalGraphDataset(Dataset):
    def __init__(
        self,
        graph_dir:  str,
        split_file: str,
        device=None,
        label_positions: dict = None,
    ):
        self.graph_dir       = graph_dir
        self.device          = device
        self.label_positions = label_positions

        self.ids = self._load_split(split_file)

        valid_ids = []
        missing_ids = []
        for pid in self.ids:
            graph_path = os.path.join(self.graph_dir, f"{pid}.pt")
            if os.path.exists(graph_path):
                valid_ids.append(pid)
            else:
                missing_ids.append(pid)

        self.ids = valid_ids

        if missing_ids:
            print(f"[HierarchicalGraphDataset] {len(missing_ids)} id(s) from "
                  f"{split_file} have no graph in {graph_dir}, excluded "
                  f"(e.g. {missing_ids[:5]})")
        print(f"[HierarchicalGraphDataset] Loaded {len(self.ids)} complexes "
              f"from {split_file}")

    def _load_split(self, split_file: str):
        ids = []
        with open(split_file) as f:
            for line in f:
                # NOTE: no .lower() -- record ids are case-sensitive
                # (e.g. '5dv7_C' vs '5dv7_c' are different files on disk)
                pid = line.strip().split()[0]
                if pid:
                    ids.append(pid)
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid        = self.ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{pid}.pt")

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)

        # ── compute labels on the fly for site-format (Metal) graphs ──
        if self.label_positions is not None and getattr(graph, "residue_id", None) is not None:
            positions = self.label_positions.get(pid, set())
            graph.residue_labels = _labels_from_positions(graph.residue_id, positions)

        # ── move to device ────────────────────────────
        if self.device is not None:
            graph.to_torch(self.device)

        return {
            "graph": graph,
            "id":    pid,
        }