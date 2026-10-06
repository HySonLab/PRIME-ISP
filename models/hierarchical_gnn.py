from torch import nn
import torch
import os
import sys

import argparse
import yaml

import scipy.sparse as sp
import numpy as np
from torch_scatter import scatter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def sparse_mm_scatter(sparse_matrix, H):
    """
    Stable replacement for torch.sparse.mm using torch_scatter.
    sparse_matrix: sparse COO tensor (M, N)
    H:             dense tensor (N, D)
    Returns:       dense tensor (M, D)
    """
    sparse_matrix = sparse_matrix.coalesce()
    row = sparse_matrix.indices()[0]  # (E,)
    col = sparse_matrix.indices()[1]  # (E,)
    val = sparse_matrix.values()      # (E,)

    msg = H[col] * val.unsqueeze(-1)  # (E, D)

    out = scatter(
        msg,
        row,
        dim=0,
        dim_size=sparse_matrix.size(0),
        reduce="sum"
    )  # (M, D)

    return out

class GraphBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, n_relations=1):
        super().__init__()

        self.n_relations = n_relations

        self.norm        = nn.LayerNorm(dim)
        self.linear_self = nn.Linear(dim, dim)

        self.linear_neigh = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(n_relations)
        ])

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        """
        A: single sparse matrix  (n_relations=1)
           or list of sparse matrices (n_relations=2)
        """
        H = self.norm(X)

        if isinstance(A, list):
            msg = self.linear_self(H)
            for A_r, linear_r in zip(A, self.linear_neigh):
                msg = msg + linear_r(sparse_mm_scatter(A_r, H))
        else:
            msg = self.linear_self(H) + self.linear_neigh[0](
                sparse_mm_scatter(A, H)
            )

        H = self.dropout(msg)
        X = X + H
        X = X + self.ff(self.norm2(X))

        return X

class GatedFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        # Gate sees both z and context
        self.gate   = nn.Linear(dim * 2, dim)
        self.update = nn.Linear(dim, dim)
        self.norm   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, context):

        # Gate depends on both current state and incoming signal
        g = torch.sigmoid(self.gate(torch.cat([z, context], dim=-1)))
        u = self.dropout(self.update(context))

        return self.norm(z + g * u)
    
class HierarchicalLayer(nn.Module):
    def __init__(self, dims, active_levels, direction="bidirectional"):
        """
        Args:
            direction: "bidirectional" | "bottom_up_only" | "top_down_only"
        """
        super().__init__()
        self.active_levels = active_levels
        self.direction     = direction

        self.gnns = nn.ModuleDict({
            level: GraphBlock(
                dims[level],
                n_relations=2 if level in ("atom", "residue") else 1
            )
            for level in active_levels
        })

        self.proj_up   = nn.ModuleDict()
        self.proj_down = nn.ModuleDict()
        self.fuse_up   = nn.ModuleDict()
        self.fuse_down = nn.ModuleDict()

        transitions = [
            ("surface", "atom",    "surface_to_atom"),
            ("atom",    "residue", "atom_to_residue"),
            ("residue", "sse",     "residue_to_sse"),
            ("sse",     "protein", "sse_to_protein"),
        ]

        for lower, upper, _ in transitions:
            if lower in active_levels and upper in active_levels:
                # only build needed modules based on direction
                if direction in ("bidirectional", "bottom_up_only"):
                    self.proj_up[f"{lower}_{upper}"] = nn.Linear(dims[lower], dims[upper])
                    self.fuse_up[upper]               = GatedFusion(dims[upper])

                if direction in ("bidirectional", "top_down_only"):
                    self.proj_down[f"{upper}_{lower}"] = nn.Linear(dims[upper], dims[lower])
                    self.fuse_down[lower]               = GatedFusion(dims[lower])

    def forward(self, graph, H):
        # ==================================================
        # 1. Intra-level GNN
        # ==================================================
        for level in self.active_levels:
            A        = getattr(graph, level).A
            H[level] = self.gnns[level](H[level], A)

        # ==================================================
        # 2. Bottom-Up (skip if top_down_only)
        # ==================================================
        if self.direction in ("bidirectional", "bottom_up_only"):
            transitions_up = [
                ("surface", "atom",    "surface_to_atom_T"),
                ("atom",    "residue", "atom_to_residue_T"),
                ("residue", "sse",     "residue_to_sse_T"),
                ("sse",     "protein", "sse_to_protein_T"),
            ]
            for lower, upper, partition_key in transitions_up:
                if lower not in H or upper not in H:
                    continue
                Pi_T     = graph.partitions[partition_key]
                msg      = sparse_mm_scatter(Pi_T, H[lower])
                msg      = self.proj_up[f"{lower}_{upper}"](msg)
                H[upper] = self.fuse_up[upper](H[upper], msg)

        # ==================================================
        # 3. Top-Down (skip if bottom_up_only)
        # ==================================================
        if self.direction in ("bidirectional", "top_down_only"):
            transitions_down = [
                ("sse",     "protein", "sse_to_protein"),
                ("residue", "sse",     "residue_to_sse"),
                ("atom",    "residue", "atom_to_residue"),
                ("surface", "atom",    "surface_to_atom"),
            ]
            for lower, upper, partition_key in transitions_down:
                if lower not in H or upper not in H:
                    continue
                Pi       = graph.partitions[partition_key]
                msg      = sparse_mm_scatter(Pi, H[upper])
                msg      = self.proj_down[f"{upper}_{lower}"](msg)
                H[lower] = self.fuse_down[lower](H[lower], msg)

        return H


class HierarchicalGNN(nn.Module):
    def __init__(
        self,
        input_dims,
        active_levels,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1,
        direction="bidirectional",
    ):
        """
        Args:
            direction: "bidirectional"   — full bottom-up + top-down (default)
                       "bottom_up_only"  — only bottom-up aggregation
                       "top_down_only"   — only top-down propagation
        """
        super().__init__()
        self.active_levels = active_levels
        self.direction     = direction

        assert direction in ("bidirectional", "bottom_up_only", "top_down_only"), \
            f"Unknown direction: {direction}"

        self.input_proj = nn.ModuleDict({
            level: nn.Linear(input_dims[level], hidden_dim)
            for level in active_levels
        })
        self.input_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in active_levels
        })
        self.dropout = nn.Dropout(dropout)

        dims = {level: hidden_dim for level in active_levels}
        self.layers = nn.ModuleList([
            HierarchicalLayer(dims, active_levels, direction=direction)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in active_levels
        })

    def forward(self, graph):
        H = {}
        for level in self.active_levels:
            x        = getattr(graph, level).X
            x        = self.input_proj[level](x)
            x        = self.input_norm[level](x)
            x        = torch.relu(x)
            x        = self.dropout(x)
            H[level] = x

        for layer in self.layers:
            H_new = layer(graph, H)
            H     = {level: H[level] + H_new[level] for level in H}

        for level in H:
            H[level] = self.final_norm[level](H[level])

        return H
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical Protein GNN"
    )

    parser.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()
    
    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    hier_cfg = config["hierarchical"]
    input_dims = hier_cfg["input_dims"]

    model = HierarchicalGNN(
        input_dims=input_dims,
        hidden_dim=hier_cfg["hidden_dim"],
        n_layers=hier_cfg["n_layers"]
    )