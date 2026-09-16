from torch import nn
import torch
import torch.nn.functional as F
from .emnn import EMNN


class EMNN_AutoEncoder(nn.Module):
    """
    Denoising model for surface graphs using EMNN.
    """

    def __init__(self,
                 in_node_nf,
                 hidden_nf,
                 latent_dim,
                 in_edge_nf=0,
                 n_layers=4):

        super().__init__()

        # ----------------------------
        # Encoder (EMNN)
        # ----------------------------
        self.encoder = EMNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=latent_dim,
            in_edge_nf=in_edge_nf,
            n_layers=n_layers
        )

        # ----------------------------
        # Noise Decoder
        # ----------------------------
        self.coord_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 3)
        )

    def forward(self,
                h,
                x_noisy,
                edge_index,
                face_index,
                edge_attr=None):
        """
        Predict noise (residual) for denoising.

        Returns:
            pred_noise : (N, 3)
            x_updated  : EMNN-updated coordinates
        """

        # Encode
        z, _ = self.encoder(
            h,
            x_noisy,
            edge_index,
            face_index,
            edge_attr
        )

        # Stabilize latent
        z = F.normalize(z, dim=-1)

        # Predict noise
        pred_noise = self.coord_decoder(z)

        return pred_noise