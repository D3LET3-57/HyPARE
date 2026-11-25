"""
Exact CLRS MPNN Processor — PyTorch implementation matching original CLRS (JAX/Haiku).

Reference: clrs/_src/processors.py (PGN class)

Key design (matching original CLRS):
    z = concat[node_features, hidden_state]  # Per-node features
    msg_1 = Linear(z)           # Per-node sender message
    msg_2 = Linear(z)           # Per-node receiver message
    msg_e = Linear(edge_fts)    # Per-edge message
    msg_g = Linear(graph_fts)   # Per-graph message (broadcast)

    msgs = msg_1[:, None] + msg_2[:, :, None] + msg_e + msg_g[None, None]
    msgs = MLP([hidden, hidden])(ReLU(msgs))  # Optional MLP on messages
    aggregated = max(msgs * adj_mask, dim=receiver)  # Max aggregation

    h_1 = Linear(z)             # Per-node update
    h_2 = Linear(aggregated)    # Message update
    ret = ReLU(h_1 + h_2)       # Combined with activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


BIG_NUMBER = 1e6


class ExactCLRSMpnn(nn.Module):
    """Exact CLRS MPNN processor matching original PGN implementation.

    This is the processor used in the official CLRS benchmark.
    It performs one round of message passing with:
    - Messages from [node_features, hidden_state] per node
    - Edge features contribute per-edge messages
    - Graph features contribute global messages
    - Max aggregation over neighbors
    - Update via Linear(z) + Linear(aggregated) + ReLU
    """

    def __init__(self, hidden_dim: int, msg_mlp_sizes: list = None,
                 use_full_adjacency: bool = False, use_ln: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_full_adjacency = use_full_adjacency

        # z = concat([node_fts, hidden]) -> [B, N, hidden + hidden]
        z_input_dim = hidden_dim + hidden_dim
        mid_size = hidden_dim

        # Per-node message functions (m_1, m_2 in original CLRS)
        self.msg_linear_1 = nn.Linear(z_input_dim, mid_size)
        self.msg_linear_2 = nn.Linear(z_input_dim, mid_size)

        # Per-edge message function (m_e in original CLRS)
        self.msg_edge = nn.Linear(hidden_dim, mid_size)

        # Per-graph message function (m_g in original CLRS)
        self.msg_graph = nn.Linear(hidden_dim, mid_size)

        # Optional MLP on messages (msgs_mlp_sizes in original CLRS)
        # Original: msgs = hk.nets.MLP(sizes)(jax.nn.relu(msgs))
        # Haiku MLP activates hidden layers, but not the final layer by default.
        if msg_mlp_sizes is not None:
            layers = []
            in_dim = mid_size
            for index, size in enumerate(msg_mlp_sizes):
                layers.append(nn.Linear(in_dim, size))
                if index < len(msg_mlp_sizes) - 1:
                    layers.append(nn.ReLU())
                in_dim = size
            self.msg_mlp = nn.Sequential(*layers)
        else:
            self.msg_mlp = None

        # Update functions (o1, o2 in original CLRS)
        self.update_linear_1 = nn.Linear(z_input_dim, hidden_dim)
        self.update_linear_2 = nn.Linear(mid_size, hidden_dim)

        # Original CLRS applies LayerNorm after the update activation.
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_ln else None

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden):
        """One step of message passing following original CLRS.

        Args:
            node_fts: [batch, num_nodes, hidden_dim] - encoded node features
            edge_fts: [batch, num_nodes, num_nodes, hidden_dim] - encoded edge features
            graph_fts: [batch, hidden_dim] - encoded graph features
            adj_mat: [batch, num_nodes, num_nodes] - adjacency matrix
            hidden: [batch, num_nodes, hidden_dim] - current hidden state

        Returns:
            new_hidden: [batch, num_nodes, hidden_dim] - updated hidden state
        """
        batch_size, num_nodes, _ = hidden.shape

        # z = concat([node_fts, hidden])  [B, N, hidden + hidden]
        z = torch.cat([node_fts, hidden], dim=-1)

        # === Compute messages ===
        msg_1 = self.msg_linear_1(z)  # [B, N, mid_size]
        msg_2 = self.msg_linear_2(z)  # [B, N, mid_size]

        # Combine: msg_1[:, None] + msg_2[:, :, None]
        msgs = msg_1.unsqueeze(1) + msg_2.unsqueeze(2)  # [B, N, N, mid_size]

        # Add edge messages
        msg_e = self.msg_edge(edge_fts)  # [B, N, N, mid_size]
        msgs = msgs + msg_e

        # Add graph messages (broadcast to all node pairs)
        msg_g = self.msg_graph(graph_fts)  # [B, mid_size]
        msgs = msgs + msg_g.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, mid_size]

        # === Optional MLP on messages ===
        if self.msg_mlp is not None:
            msgs = self.msg_mlp(F.relu(msgs))

        # === Aggregate messages from neighbors (max aggregation) ===
        # msgs: [B, N, N, mid_size] where msgs[b, i, j] = message from i to j
        # Aggregate over sender dimension (dim=1) to get messages TO each node
        if self.use_full_adjacency:
            adj_mat = torch.ones_like(adj_mat)
        mask = adj_mat.unsqueeze(-1)  # [B, N, N, 1]
        masked_messages = torch.where(
            mask > 0,
            msgs,
            torch.tensor(-BIG_NUMBER, dtype=msgs.dtype, device=msgs.device)
        )
        aggregated, _ = masked_messages.max(dim=1)  # [B, N, mid_size]

        # Handle case where node has no neighbors
        aggregated = torch.where(
            aggregated <= -BIG_NUMBER / 2,
            torch.zeros_like(aggregated),
            aggregated
        )

        # === Update: ret = o1(z) + o2(aggregated) ===
        h_1 = self.update_linear_1(z)       # [B, N, hidden_dim]
        h_2 = self.update_linear_2(aggregated)  # [B, N, hidden_dim]

        new_hidden = h_1 + h_2
        new_hidden = F.relu(new_hidden)
        if self.layer_norm is not None:
            new_hidden = self.layer_norm(new_hidden)

        return new_hidden


class ExactCLRSGgnn(nn.Module):
    """GGNN-style recurrent processor with the exact CLRS tensor contract."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_encoder = nn.Linear(1, hidden_dim)
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden):
        sender = hidden.unsqueeze(2).expand(-1, -1, hidden.size(1), -1)
        edge_summary = edge_fts.mean(dim=-1, keepdim=True)
        edge_state = self.edge_encoder(edge_summary)
        messages = self.message(torch.cat([sender, edge_state], dim=-1))
        mask = adj_mat.unsqueeze(-1)
        messages = messages.masked_fill(mask == 0, -1e6)
        aggregated = messages.max(dim=1).values
        aggregated = torch.where(aggregated < -1e5, torch.zeros_like(aggregated), aggregated)
        return self.gru(
            aggregated.reshape(-1, aggregated.size(-1)),
            hidden.reshape(-1, hidden.size(-1))
        ).reshape_as(hidden)
