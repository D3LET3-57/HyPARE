"""
Exact CLRS Decoder — PyTorch implementation matching original CLRS (JAX/Haiku).

Reference: clrs/_src/decoders.py

Key design for POINTER decoding (matching original CLRS):
    p_1 = Linear(h_t)           # [B, N, 3*H] -> [B, N, H] - query
    p_2 = Linear(h_t)           # [B, N, 3*H] -> [B, N, H] - key for receiver
    p_3 = Linear(edge_fts)      # [B, N, N, H] -> [B, N, N, H] - edge contribution

    p_e = p_2[:, None, :, :] + p_3           # [B, N, N, H] (receiver key + edge)
    p_m = max(p_1[:, :, None, :],             # [B, N, 1, H] (query)
              p_e.transpose(1, 2))            # [B, N, N, H] transposed

    preds = Linear(p_m)                       # [B, N, N, H] -> [B, N, N, 1] -> [B, N, N]

IMPORTANT: p_1, p_2 take h_t (3*hidden_dim) as input, while p_3 takes edge_fts (hidden_dim).
This matches the original CLRS construct_decoders where all decoders are Linear(hidden_dim)
but they receive different inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExactCLRSPointerDecoder(nn.Module):
    """Exact CLRS pointer decoder matching original implementation.

    Produces [batch, num_nodes, num_nodes] scores where
    scores[b, i, j] measures compatibility of node i pointing to node j.

    Uses edge features and max operation (matching original CLRS).

    Args:
        hidden_dim: The net's hidden dimension (128)
        decoder_input_dim: The decoder input dimension (3 * hidden_dim = 384)
    """

    def __init__(self, hidden_dim: int, decoder_input_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_input_dim = decoder_input_dim

        # p_1, p_2: take h_t (decoder_input_dim) -> hidden_dim
        self.p_1 = nn.Linear(decoder_input_dim, hidden_dim)   # Query from h_t
        self.p_2 = nn.Linear(decoder_input_dim, hidden_dim)   # Key from h_t (receiver)
        # p_3: takes edge_fts (hidden_dim) -> hidden_dim
        self.p_3 = nn.Linear(hidden_dim, hidden_dim)          # Edge contribution
        # Final: hidden_dim -> 1
        self.p_final = nn.Linear(hidden_dim, 1)               # Final projection to scalar

    def forward(self, h_t: torch.Tensor, edge_fts: torch.Tensor,
                adj_mat: torch.Tensor = None) -> torch.Tensor:
        """Decode pointer scores.

        Args:
            h_t: [batch, num_nodes, decoder_input_dim] - decoder input (3*hidden_dim)
            edge_fts: [batch, num_nodes, num_nodes, hidden_dim] - edge features
            adj_mat: [batch, num_nodes, num_nodes] - adjacency matrix for masking

        Returns:
            scores: [batch, num_nodes, num_nodes] pointer score matrix
        """
        # p_1: query [B, N, 3*H] -> [B, N, H]
        p_1 = self.p_1(h_t)
        # p_2: key for receiver [B, N, 3*H] -> [B, N, H]
        p_2 = self.p_2(h_t)
        # p_3: edge contribution [B, N, N, H] -> [B, N, N, H]
        p_3 = self.p_3(edge_fts)

        # p_e = p_2[:, :, None, :] + p_3  -> [B, N, N, H]
        # p_2 expanded at the receiver axis, matching jnp.expand_dims(p_2, -2).
        p_e = p_2.unsqueeze(2) + p_3

        # p_m = max(p_1[:, :, None, :], p_e.transpose(1, 2))
        # p_1 expanded: [B, N, 1, H] (query dimension)
        # p_e transposed: [B, N, N, H] -> [B, N, N, H] (swap source/receiver)
        p_m = torch.maximum(
            p_1.unsqueeze(2),                    # [B, N, 1, H]
            p_e.transpose(1, 2)                  # [B, N, N, H]
        )

        # Final projection: [B, N, N, H] -> [B, N, N, 1] -> [B, N, N]
        scores = self.p_final(p_m).squeeze(-1)

        # CRITICAL: Mask predictions where there's no edge (inf_bias)
        # This matches original CLRS _decode_node_fts:
        #   preds = jnp.where(adj_mat > 0.5, preds, min(-1, per_batch_min - 1))
        if adj_mat is not None:
            per_batch_min = scores.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            mask_value = torch.minimum(
                torch.tensor(-1.0, device=scores.device),
                per_batch_min - 1.0
            )
            scores = torch.where(adj_mat > 0.5, scores, mask_value)

        return scores


class ExactCLRSMaskDecoder(nn.Module):
    """Exact CLRS mask/categorical decoder."""

    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """Decode mask/categorical logits.

        Args:
            h_t: [batch, num_nodes, hidden_dim]

        Returns:
            logits: [batch, num_nodes, num_classes]
        """
        return self.linear(h_t)


class ExactCLRSDecoder(nn.Module):
    """Combined decoder for hints and outputs following original CLRS.

    Args:
        hidden_dim: The net's hidden dimension (128)
        decoder_input_dim: The decoder input dimension (3 * hidden_dim = 384)
        hint_type: 'pointer', 'mask', 'mask_one', or 'categorical'
        output_type: 'pointer', 'mask', 'mask_one', or 'categorical'
        num_hint_classes: Number of classes for non-pointer hints
        num_output_classes: Number of classes for non-pointer outputs
    """

    def __init__(self, hidden_dim: int = 128,
                 decoder_input_dim: int = 384,
                 hint_type: str = 'pointer',
                 output_type: str = 'pointer',
                 num_hint_classes: int = 2,
                 num_output_classes: int = 2):
        super().__init__()
        self.hint_type = hint_type
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.decoder_input_dim = decoder_input_dim

        # Hint decoder
        if hint_type == 'pointer':
            self.hint_decoder = ExactCLRSPointerDecoder(hidden_dim, decoder_input_dim)
        else:
            self.hint_decoder = ExactCLRSMaskDecoder(decoder_input_dim, num_hint_classes)

        # Output decoder
        if output_type == 'pointer':
            self.output_decoder = ExactCLRSPointerDecoder(hidden_dim, decoder_input_dim)
        else:
            self.output_decoder = ExactCLRSMaskDecoder(decoder_input_dim, num_output_classes)

        # Mask decoder for mask-type hints (e.g., reach_h in BFS)
        # Matches original CLRS: Linear(decoder_input_dim, 1) for MASK type
        self.mask_decoder = ExactCLRSMaskDecoder(decoder_input_dim, 1)

    def decode_hint(self, h_t: torch.Tensor, edge_fts: torch.Tensor,
                    adj_mat: torch.Tensor = None) -> torch.Tensor:
        """Decode intermediate hint.

        Args:
            h_t: [batch, num_nodes, decoder_input_dim]
            edge_fts: [batch, num_nodes, num_nodes, hidden_dim]
            adj_mat: [batch, num_nodes, num_nodes] adjacency for masking

        Returns:
            hint_pred: [batch, num_nodes, num_nodes] for pointer
                      [batch, num_nodes, num_classes] for mask
        """
        if self.hint_type == 'pointer':
            return self.hint_decoder(h_t, edge_fts, adj_mat)
        return self.hint_decoder(h_t)

    def decode_mask_hint(self, h_t: torch.Tensor) -> torch.Tensor:
        """Decode mask hint (reach_h) from h_t.

        Matches original CLRS _decode_node_fts for MASK type:
            preds = squeeze(decoders[0](h_t), -1)  -> [B, N]

        Args:
            h_t: [batch, num_nodes, decoder_input_dim]

        Returns:
            mask_pred: [batch, num_nodes] mask logits
        """
        # Use a dedicated mask decoder (shared with hint_decoder for mask type)
        if hasattr(self, 'mask_decoder'):
            return self.mask_decoder(h_t).squeeze(-1)  # [B, N, 1] -> [B, N]
        # Fallback: reuse hint_decoder if it's a mask type
        return self.hint_decoder(h_t)

    def decode_output(self, h_t: torch.Tensor, edge_fts: torch.Tensor,
                      adj_mat: torch.Tensor = None) -> torch.Tensor:
        """Decode final output.

        Args:
            h_t: [batch, num_nodes, decoder_input_dim]
            edge_fts: [batch, num_nodes, num_nodes, hidden_dim]
            adj_mat: [batch, num_nodes, num_nodes] adjacency for masking

        Returns:
            output_pred: [batch, num_nodes, num_nodes] for pointer
                        [batch, num_nodes, num_classes] for mask
        """
        if self.output_type == 'pointer':
            return self.output_decoder(h_t, edge_fts, adj_mat)
        return self.output_decoder(h_t)
