# model.py
import torch
from torch import nn
from .hierarchical_gnn import HierarchicalGNN

import torch
from torch import nn
from .hierarchical_gnn import HierarchicalGNN
from .classification_head import MLP_Head

from esm import pretrained
from .egnn import EGNN
from .emnn import EMNN
import torch.nn.functional as F
from torch_scatter import scatter

class PRIME(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dims,
        active_levels,
        readout_level,
        hidden_dim=512,
        encoder_layers=3,
        head_hidden_dim=512,
        head_layers=3,
        dropout=0.3,
        task_level="graph",
    ):
        super().__init__()

        self.readout_level = readout_level
        self.task_level    = task_level

        self.encoder = HierarchicalGNN(
            input_dims=input_dims,
            active_levels=active_levels,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers
        )

        self.head = MLP_Head(
            in_dim=hidden_dim,
            out_dim=num_classes,
            hidden_dims=[head_hidden_dim] * head_layers,
            activations=["gelu"] * head_layers + ["identity"],
            dropout=dropout,
            skip=True
        )

    def forward(self, graph):
        H         = self.encoder(graph)
        embedding = H[self.readout_level]

        if self.task_level == "node":
            return self.head(embedding)          # (N_res, num_classes)

        embedding = embedding.mean(dim=0, keepdim=True)
        return self.head(embedding)              # (1, num_classes)
    
class LevelAttentionReadout(nn.Module):
    """
    Adaptive cross-attention readout over all active hierarchical levels.
    A single learnable query attends over mean-pooled level representations
    to produce a unified task embedding.
    """
    def __init__(self, hidden_dim, active_levels):
        super().__init__()

        self.active_levels = active_levels

        self.query = nn.Parameter(torch.randn(1, hidden_dim))

        self.key_proj = nn.ModuleDict({
            level: nn.Linear(hidden_dim, hidden_dim)
            for level in active_levels
        })
        self.val_proj = nn.ModuleDict({
            level: nn.Linear(hidden_dim, hidden_dim)
            for level in active_levels
        })

        self.scale = hidden_dim ** -0.5
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, H):
        """
        H: dict of {level: (N_level, hidden_dim)}
        Returns:
            embedding:      (1, hidden_dim)
            attn_weights:   (1, L) — interpretable level importance
        """
        pooled = {
            level: H[level].mean(dim=0, keepdim=True)
            for level in self.active_levels
        }

        keys = torch.cat([
            self.key_proj[level](pooled[level])
            for level in self.active_levels
        ], dim=0)

        values = torch.cat([
            self.val_proj[level](pooled[level])
            for level in self.active_levels
        ], dim=0)
        
        attn_scores  = (self.query @ keys.T) * self.scale   # (1, L)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (1, L)

        embedding = attn_weights @ values                    # (1, hidden_dim)
        embedding = self.norm(embedding)

        return embedding, attn_weights

class PRIME_CrossAttention(nn.Module):
    """
    PRIME variant with adaptive multiscale readout via cross-attention.
    Replaces fixed readout level with learned level-attention mechanism.
    """
    def __init__(
        self,
        num_classes,
        input_dims,
        active_levels,
        hidden_dim=128,
        encoder_layers=2,
        head_hidden_dim=128,
        head_layers=2,
        dropout=0.3,
        task_level="graph",
    ):
        super().__init__()

        self.active_levels = active_levels
        self.task_level    = task_level

        self.encoder = HierarchicalGNN(
            input_dims=input_dims,
            active_levels=active_levels,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers
        )

        self.readout = LevelAttentionReadout(hidden_dim, active_levels)

        self.head = MLP_Head(
            in_dim=hidden_dim,
            out_dim=num_classes,
            hidden_dims=[head_hidden_dim] * head_layers,
            activations=["gelu"] * head_layers + ["identity"],
            dropout=dropout,
            skip=True
        )

    def forward(self, graph, return_attn=False):
        """
        Args:
            graph:       HierarchicalProteinGraph
            return_attn: if True, also return attention weights for analysis

        Returns:
            logits:       (1, num_classes)
            attn_weights: (1, L) — only if return_attn=True
        """
        H = self.encoder(graph)

        embedding, attn_weights = self.readout(H)  # (1, hidden_dim), (1, L)

        logits = self.head(embedding)              # (1, num_classes)

        if return_attn:
            return logits, attn_weights

        return logits