"""HyPARE processors on top of the CLRS-style MPNN backbone.

The recurrent branch is node-level: BFS maintains different algorithmic states at
different nodes, so broadcasting one graph-level recurrent vector would erase the
very distinctions needed by pointer decoding.
"""

import torch
import torch.nn as nn

from .mpnn import ExactCLRSMpnn


class _NodeRecurrentBackbone(nn.Module):
    def __init__(self, hidden_dim: int, use_ln: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mpnn = ExactCLRSMpnn(
            hidden_dim=hidden_dim,
            msg_mlp_sizes=[hidden_dim, hidden_dim],
            use_full_adjacency=True,
            use_ln=use_ln,
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def branches(self, node_fts, edge_fts, graph_fts, adj_mat, hidden):
        parallel = self.mpnn(node_fts, edge_fts, graph_fts, adj_mat, hidden)
        recurrent = self.gru(
            parallel.reshape(-1, self.hidden_dim),
            hidden.reshape(-1, self.hidden_dim),
        ).reshape_as(hidden)
        return parallel, recurrent


class ExactCLRSMpnnGRU(_NodeRecurrentBackbone):
    """Fixed node-level mixture used as the recurrent ablation."""

    def __init__(self, hidden_dim: int, alpha: float = 0.5,
                 use_ln: bool = True):
        super().__init__(hidden_dim, use_ln=use_ln)
        self.alpha = alpha

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden,
                gru_state=None):
        parallel, recurrent = self.branches(
            node_fts, edge_fts, graph_fts, adj_mat, hidden)
        mixed = (1.0 - self.alpha) * parallel + self.alpha * recurrent
        return mixed, None


class ExactCLRSHyPARE(_NodeRecurrentBackbone):
    """Adaptive hybrid of parallel message passing and node-level recurrence.

    M_t = MPNN(H_t, X, E)
    R_t = GRU(M_t, H_t)
    alpha_t = sigmoid(g(mean_nodes(H_t)))
    H_{t+1} = (1-alpha_t) M_t + alpha_t R_t

    Args:
        hidden_dim: hidden dimension.
        gate_init: initial bias of the gate pre-sigmoid layer. More negative
            starts closer to pure MPNN; less negative enables recurrence earlier.
    """

    def __init__(self, hidden_dim: int, gate_init: float = -2.0,
                 use_ln: bool = True):
        super().__init__(hidden_dim, use_ln=use_ln)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_network[-2].bias, gate_init)

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden,
                gru_state=None):
        parallel, recurrent = self.branches(
            node_fts, edge_fts, graph_fts, adj_mat, hidden)
        gate = self.gate_network(hidden.mean(dim=1))
        alpha = gate.unsqueeze(-1)
        mixed = (1.0 - alpha) * parallel + alpha * recurrent
        return mixed, gate, None
