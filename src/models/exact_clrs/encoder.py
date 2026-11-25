"""
Exact CLRS Encoder — PyTorch implementation matching original CLRS (JAX/Haiku).

Reference: clrs/_src/encoders.py

Key design:
- Each input feature gets a SINGLE Linear(1, hidden_dim) layer
- Encoders are shared between inputs and hints (same weights)
- Features are accumulated into node_fts, edge_fts, graph_fts
- Adjacency matrix is dynamically updated from pointer features

Note: The CLRS adapter combines multiple features into a single tensor.
We split them along the feature dimension and process each separately
to match the original CLRS behavior.

CRITICAL: This must be a nn.Module so that encoder parameters are
registered with the parent model and updated by the optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExactCLREncoder(nn.Module):
    """Exact CLRS encoder matching original implementation.

    Creates one Linear(1, hidden_dim) encoder per input/hint feature.
    These encoders are shared between inputs and hints.

    Uses nn.ModuleDict to ensure all encoder parameters are properly
    registered with the parent model.
    """

    def __init__(self, hidden_dim: int, node_feat_dim: int,
                 edge_feat_dim: int, graph_feat_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        names = (
            [f'node_{i}' for i in range(node_feat_dim)]
            + [f'edge_{i}' for i in range(edge_feat_dim)]
            + [f'graph_{i}' for i in range(graph_feat_dim)]
            + ['pi_h_ptr', 'reach_h_mask']
        )
        # Every parameter exists before the optimizer is constructed.
        self.encoders = nn.ModuleDict({
            name: nn.Linear(1, hidden_dim) for name in names
        })

    def get_encoder(self, name: str, device: torch.device) -> nn.Linear:
        """Return a pre-registered per-feature encoder."""
        if name not in self.encoders:
            raise KeyError(f'Encoder {name!r} was not registered at model construction')
        return self.encoders[name]

    def encode_features(self, node_features, edge_features, graph_features,
                        adj_mat, nb_nodes, device):
        """Encode input/hint features into node/edge/graph features.

        Matches original CLRS accumulation pattern:
        - node_fts = sum of encoded node features
        - edge_fts = sum of encoded edge features
        - graph_fts = sum of encoded graph features
        - adj_mat updated from pointer features

        Args:
            node_features: [batch, num_nodes, node_feat_dim] or None
            edge_features: [batch, num_nodes, num_nodes, edge_feat_dim] or None
            graph_features: [batch, graph_feat_dim] or None
            adj_mat: [batch, num_nodes, num_nodes] adjacency matrix
            nb_nodes: number of nodes
            device: torch device

        Returns:
            node_fts: [batch, num_nodes, hidden_dim]
            edge_fts: [batch, num_nodes, num_nodes, hidden_dim]
            graph_fts: [batch, hidden_dim]
            adj_mat: updated adjacency matrix
        """
        batch_size = adj_mat.shape[0]

        node_fts = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device)
        edge_fts = torch.zeros(batch_size, nb_nodes, nb_nodes, self.hidden_dim, device=device)
        graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device)

        # === Encode node features ===
        # Original CLRS: each node feature gets its own Linear(1, hidden)
        # The adapter combines them as [B, N, feat_dim], so we split
        if node_features is not None and node_features.shape[-1] > 0:
            feat_dim = node_features.shape[-1]
            for i in range(feat_dim):
                encoder = self.get_encoder(f'node_{i}', device)
                # [B, N, 1] -> [B, N, hidden]
                feat = node_features[:, :, i:i+1]
                node_fts = node_fts + encoder(feat)

        # === Encode edge features ===
        if edge_features is not None and edge_features.shape[-1] > 0:
            feat_dim = edge_features.shape[-1]
            for i in range(feat_dim):
                encoder = self.get_encoder(f'edge_{i}', device)
                # [B, N, N, 1] -> [B, N, N, hidden]
                feat = edge_features[:, :, :, i:i+1]
                edge_fts = edge_fts + encoder(feat)

        # === Encode graph features ===
        if graph_features is not None and graph_features.shape[-1] > 0:
            feat_dim = graph_features.shape[-1]
            for i in range(feat_dim):
                encoder = self.get_encoder(f'graph_{i}', device)
                # [B, 1] -> [B, hidden]
                feat = graph_features[:, i:i+1]
                graph_fts = graph_fts + encoder(feat)

        return node_fts, edge_fts, graph_fts, adj_mat
