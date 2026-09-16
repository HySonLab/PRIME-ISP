import torch
import torch.nn as nn
from typing import List

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")

class LinearSkipBlock(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        activations: List[str],
        out_dim: int,
        dropout: float,
        skip_type: str = "sum"
    ):
        super().__init__()

        self.skip_type = skip_type

        layers = []
        layers.append(nn.LazyLinear(hidden_dims[0]))
        layers.append(get_activation(activations[0]))
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(get_activation(activations[i + 1]))
            layers.append(nn.Dropout(dropout))

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        prev = x
        x = self.hidden(x)

        if self.skip_type == "sum" and x.shape == prev.shape:
            x = x + prev
        elif self.skip_type == "concat":
            x = torch.cat([x, prev], dim=-1)

        return self.output(x)

class MLP_Head(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activations: List[str],
        dropout: float = 0.1,
        skip: bool = False,
    ):
        super().__init__()

        assert len(activations) == len(hidden_dims) + 1, \
            "Need len(hidden_dims) + 1 activations"

        self.input_proj = nn.Identity()
        if in_dim != hidden_dims[0]:
            self.input_proj = nn.Linear(in_dim, hidden_dims[0])

        if skip:
            self.layers = LinearSkipBlock(
                hidden_dims,
                activations,
                out_dim,
                dropout,
                skip_type="sum"
            )
        else:
            layers = []

            layers.append(get_activation(activations[0]))
            layers.append(nn.Dropout(dropout))

            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(get_activation(activations[i + 1]))
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dims[-1], out_dim))
            layers.append(get_activation(activations[-1]))

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.layers(x)