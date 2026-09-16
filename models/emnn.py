from torch import nn
import torch

import torch
from torch import nn


class EMNN_GCL(nn.Module):
    """
    E(3)-Equivariant Mesh Neural Network Layer
    Edge + Face message passing
    """

    def __init__(self,
                 input_nf,
                 output_nf,
                 hidden_nf,
                 edges_in_d=0,
                 act_fn=nn.SiLU(),
                 residual=True):

        super().__init__()

        self.residual = residual

        # ---------------------------
        # Edge message MLP
        # ---------------------------
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # ---------------------------
        # Face message MLP
        # ---------------------------
        self.face_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # ---------------------------
        # Node update
        # ---------------------------
        self.node_mlp = nn.Sequential(
            nn.Linear(input_nf + hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # ---------------------------
        # Coordinate updates
        # ---------------------------
        self.coord_mlp_edge = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )

        self.coord_mlp_face = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )

    # =========================================================
    # Forward
    # =========================================================
    def forward(self,
                h,
                x,
                edge_index,
                face_index,
                edge_attr=None):

        # -----------------------
        # EDGE MESSAGES
        # -----------------------
        row, col = edge_index

        coord_diff = x[row] - x[col]
        radial = (coord_diff ** 2).sum(dim=1, keepdim=True)

        if edge_attr is None:
            edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        else:
            edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        # -----------------------
        # FACE MESSAGES
        # -----------------------
        i, j, k = face_index

        vec_ji = x[j] - x[i]
        vec_ki = x[k] - x[i]

        cross = torch.cross(vec_ji, vec_ki, dim=1)
        area = torch.norm(cross, dim=1, keepdim=True)

        face_input = torch.cat([h[i], h[j] + h[k], area], dim=-1)
        m_ijk = self.face_mlp(face_input)

        # -----------------------
        # NODE UPDATE
        # -----------------------
        edge_agg = unsorted_segment_sum(m_ij, row, h.size(0))
        face_agg = unsorted_segment_sum(m_ijk, i, h.size(0))

        h_input = torch.cat([h, edge_agg, face_agg], dim=-1)
        h_out = self.node_mlp(h_input)

        if self.residual:
            h_out = h + h_out

        # -----------------------
        # COORD UPDATE
        # -----------------------
        edge_coef = self.coord_mlp_edge(m_ij)
        edge_trans = coord_diff * edge_coef
        edge_update = unsorted_segment_sum(edge_trans, row, x.size(0))

        face_coef = self.coord_mlp_face(m_ijk)
        face_trans = cross * face_coef
        face_update = unsorted_segment_sum(face_trans, i, x.size(0))

        x_out = x + edge_update + face_update

        return h_out, x_out

class EMNN(nn.Module):
    """
    Multi-layer EMNN
    """

    def __init__(self,
                 in_node_nf,
                 hidden_nf,
                 out_node_nf,
                 in_edge_nf=0,
                 device='cpu',
                 act_fn=nn.SiLU(),
                 n_layers=4,
                 residual=True):

        super().__init__()

        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                EMNN_GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual
                )
            )

        self.to(device)

    # =========================================================
    # Forward (same format as EGNN + face_index)
    # =========================================================
    def forward(self,
                h,
                x,
                edge_index,
                face_index,
                edge_attr=None):

        h = self.embedding_in(h)

        for i in range(self.n_layers):
            h, x = self._modules[f"gcl_{i}"](
                h,
                x,
                edge_index,
                face_index,
                edge_attr
            )

        h = self.embedding_out(h)

        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EMNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)