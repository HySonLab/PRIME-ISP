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
        direction="bidirectional",
    ):
        super().__init__()
 
        self.readout_level = readout_level
        self.task_level    = task_level
        self.direction      = direction
 
        self.encoder = HierarchicalGNN(
            input_dims=input_dims,
            active_levels=active_levels,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers,
            direction=direction,
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