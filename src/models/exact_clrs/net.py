"""
Exact CLRS Net — PyTorch implementation matching original CLRS (JAX/Haiku).

Reference: clrs/_src/nets.py (Net class)

This is the core encode-process-decode pipeline that:
1. Encodes inputs and hints into node/edge/graph features
2. Runs the processor for nb_msg_passing_steps
3. Decodes hints and outputs from [node_fts, hidden, nxt_hidden]

CRITICAL: Matches original CLRS hint processing exactly:
- NODE POINTER hints: encoded and accumulated into EDGE features
- Adjacency matrix updated from pointer hints
- Same encoder used for hints (by name) as inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ExactCLREncoder
from .decoder import ExactCLRSDecoder
from .mpnn import ExactCLRSMpnn, ExactCLRSGgnn
from .hypare import ExactCLRSHyPARE, ExactCLRSMpnnGRU


class ExactCLRSNet(nn.Module):
    """Exact CLRS Net matching original implementation."""

    def __init__(self, node_feat_dim: int = 1, edge_feat_dim: int = 1,
                 graph_feat_dim: int = 0, hidden_dim: int = 128,
                 nb_msg_passing_steps: int = 1,
                 hint_type: str = 'pointer',
                 output_type: str = 'pointer',
                 num_hint_classes: int = 2,
                 num_output_classes: int = 2,
                 dropout_prob: float = 0.0,
                 processor_type: str = 'mpnn',
                 gate_init: float = -2.0,
                 use_ln: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nb_msg_passing_steps = nb_msg_passing_steps
        self.hint_type = hint_type
        self.output_type = output_type
        self.dropout_prob = dropout_prob
        self.processor_type = processor_type

        # Encoder (shared between inputs and hints)
        self.encoder = ExactCLREncoder(
            hidden_dim, node_feat_dim, edge_feat_dim, graph_feat_dim)

        if processor_type == 'mpnn':
            self.processor = ExactCLRSMpnn(
                hidden_dim=hidden_dim,
                msg_mlp_sizes=[hidden_dim, hidden_dim],
                use_full_adjacency=True,
                use_ln=use_ln,
            )
        elif processor_type == 'ggnn':
            self.processor = ExactCLRSGgnn(hidden_dim=hidden_dim)
        elif processor_type == 'mpnn_gru':
            # MPNN + GRU with fixed alpha (ablation baseline)
            self.processor = ExactCLRSMpnnGRU(hidden_dim=hidden_dim, alpha=0.5,
                                              use_ln=use_ln)
        elif processor_type == 'hypare':
            # HyPARE: MPNN + GRU + adaptive gate (our proposed model)
            self.processor = ExactCLRSHyPARE(hidden_dim=hidden_dim,
                                             gate_init=gate_init,
                                             use_ln=use_ln)
        else:
            raise ValueError(f'Unknown exact processor_type: {processor_type}')

        # Decoder input dim = 3 * hidden_dim for [node_fts, hidden, nxt_hidden]
        decoder_input_dim = 3 * hidden_dim
        self.decoder = ExactCLRSDecoder(
            hidden_dim=hidden_dim,
            decoder_input_dim=decoder_input_dim,
            hint_type=hint_type,
            output_type=output_type,
            num_hint_classes=num_hint_classes,
            num_output_classes=num_output_classes
        )

    def _encode_inputs(self, node_features, edge_features, graph_features,
                       adj_mat, nb_nodes, device):
        """Encode input features matching original CLRS.

        Matches original CLRS _one_step_pred:
        1. Start with identity adjacency (self-loops)
        2. Accumulate input edges into adjacency
        3. Encode input features

        Returns:
            node_fts, edge_fts, graph_fts, adj_mat
        """
        # Start with identity (self-loops) like original CLRS
        # Then add input adjacency on top
        eye = torch.eye(nb_nodes, device=device).unsqueeze(0).expand(adj_mat.shape[0], -1, -1)
        adj_mat = ((eye + adj_mat) > 0).float()

        return self.encoder.encode_features(
            node_features, edge_features, graph_features,
            adj_mat, nb_nodes, device
        )

    def _encode_hint(self, hint_data, hint_type, nb_nodes, device, batch_size=None,
                     hint_name='hint'):
        """Encode hint matching original CLRS hint processing.

        For NODE POINTER hints (BFS case):
        1. Preprocess: indices -> one-hot [B, N, N]
        2. Update adjacency matrix from pointer
        3. Encode pointer and accumulate into EDGE features (not node!)

        For NODE MASK hints (BFS case):
        1. Encode mask and accumulate into NODE features

        This matches original CLRS accum_edge_fts / accum_node_fts behavior.

        Args:
            hint_data: raw hint tensor [B, N] indices or [B, N, N] scores
            hint_type: 'pointer', 'mask', etc.
            nb_nodes: number of nodes
            device: torch device
            batch_size: batch size (inferred from hint_data if not given)
            hint_name: feature name for encoder lookup (matches original CLRS per-feature encoders)

        Returns:
            hint_node_fts: [B, N, H] (encoded mask for mask hints, zeros for pointer)
            hint_edge_fts: [B, N, N, H] (encoded pointer for pointer hints, zeros for mask)
            hint_graph_fts: [B, H] (zeros for both)
            hint_adj: [B, N, N] adjacency update from pointer
        """
        # Infer batch_size from hint_data if available, else use provided
        if hint_data is not None and isinstance(hint_data, torch.Tensor):
            batch_size = hint_data.shape[0]
        elif batch_size is None:
            # Fallback: cannot determine batch size
            batch_size = 1

        hint_node_fts = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device)
        hint_edge_fts = torch.zeros(batch_size, nb_nodes, nb_nodes, self.hidden_dim, device=device)
        hint_graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device)
        hint_adj = torch.zeros(batch_size, nb_nodes, nb_nodes, device=device)

        # Handle None hint_data (can happen at certain steps)
        if hint_data is None or not isinstance(hint_data, torch.Tensor):
            return hint_node_fts, hint_edge_fts, hint_graph_fts, hint_adj

        if hint_type == 'pointer':
            # Preprocess: indices -> one-hot [B, N, N]
            if hint_data.dim() == 2:
                one_hot = torch.zeros(batch_size, nb_nodes, nb_nodes, device=device)
                indices = hint_data.long().clamp(0, nb_nodes - 1)
                one_hot.scatter_(2, indices.unsqueeze(-1), 1.0)
            else:
                one_hot = hint_data.float()

            # Update adjacency from pointer (matching original CLRS accum_adj_mat)
            hint_adj = ((one_hot + one_hot.transpose(1, 2)) > 0.5).float()

            # Encode pointer into EDGE features (matching original CLRS accum_edge_fts)
            # NODE POINTER -> edge_fts (NOT node_fts!)
            # Use per-feature-name encoder (matching original CLRS)
            hint_edge_fts = self.encoder.get_encoder(f'{hint_name}_ptr', device)(
                one_hot.unsqueeze(-1)
            )  # [B, N, N, 1] -> [B, N, N, H]

        else:
            # Mask/categorical hint -> node features (matching original CLRS accum_node_fts)
            if hint_data.dim() == 2:
                hint_data = hint_data.unsqueeze(-1)
            hint_node_fts = self.encoder.get_encoder(f'{hint_name}_mask', device)(hint_data)

        return hint_node_fts, hint_edge_fts, hint_graph_fts, hint_adj

    def _postprocess_hint(self, hint_pred, hint_type, hard=False):
        """Postprocess hint prediction matching original CLRS."""
        if hint_type == 'pointer':
            if hard:
                return hint_pred.argmax(dim=-1)
            return F.softmax(hint_pred, dim=-1)
        if hint_type == 'mask':
            return torch.sigmoid(hint_pred)
        return hint_pred

    def forward(self, node_features, edge_features, graph_features,
                adjacency, hint_targets=None, hint_masks=None,
                pointer_targets=None, pointer_masks=None,
                mask_targets=None, mask_hint_masks=None, lengths=None,
                num_steps=None, teacher_forcing_ratio=0.0,
                nb_nodes=None):
        """Forward pass matching original CLRS Net.

        Matches original CLRS multi-hint processing:
        - NODE POINTER hints (pi_h): encoded into edge features, decoded as pointer
        - NODE MASK hints (reach_h): encoded into node features, decoded as mask
        - Both hints processed at each step (matching original CLRS)
        """
        batch_size = node_features.size(0)
        if nb_nodes is None:
            nb_nodes = node_features.size(1)
        device = node_features.device

        if num_steps is None:
            if pointer_targets is not None:
                num_steps = pointer_targets.size(1) - 1
            elif hint_targets is not None:
                num_steps = hint_targets.size(1) - 1
            elif mask_targets is not None:
                num_steps = mask_targets.size(1) - 1
            else:
                num_steps = 10
            num_steps = max(1, num_steps)

        # === ENCODE INPUTS ===
        node_fts, edge_fts, graph_fts, adj_mat = self._encode_inputs(
            node_features, edge_features, graph_features,
            adjacency.clone(), nb_nodes, device
        )

        # === INITIAL HIDDEN STATE ===
        hidden = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device)

        # GRU separate state (carried between execution steps, like hidden)
        # Initialized to zeros, carried forward through the execution loop
        gru_state = None

        # Storage for outputs
        pointer_hint_preds = []
        mask_hint_preds = []
        output_pred = None

        # Determine if we have mask hints
        has_mask_hints = mask_targets is not None and mask_targets.size(1) > 0

        # === EXECUTION LOOP ===
        gate_values = None
        for step in range(num_steps):
            # CLRS noisy teacher forcing samples once per example and shares the
            # decision across all hint fields.
            force_mask = (
                torch.rand(batch_size, device=device) < teacher_forcing_ratio
                if step > 0 else torch.ones(batch_size, dtype=torch.bool, device=device)
            )

            # === GET POINTER HINT (pi_h) ===
            if step == 0:
                ptr_hint_data = (
                    pointer_targets[:, 0]
                    if pointer_targets is not None and pointer_targets.size(1) > 0
                    else None
                )
            else:
                predicted_ptr = self._postprocess_hint(
                    pointer_hint_preds[-1], 'pointer', hard=False)
                if pointer_targets is not None and step < pointer_targets.size(1):
                    target_ptr = F.one_hot(
                        pointer_targets[:, step].long().clamp(0, nb_nodes - 1),
                        num_classes=nb_nodes,
                    ).float()
                    ptr_hint_data = torch.where(
                        force_mask[:, None, None], target_ptr, predicted_ptr)
                else:
                    ptr_hint_data = predicted_ptr

            # === GET MASK HINT (reach_h) ===
            if has_mask_hints:
                assert mask_targets is not None
                if step == 0:
                    mask_hint_data = mask_targets[:, 0]
                else:
                    predicted_mask = self._postprocess_hint(
                        mask_hint_preds[-1], 'mask', hard=False)
                    if step < mask_targets.size(1):
                        mask_hint_data = torch.where(
                            force_mask[:, None], mask_targets[:, step], predicted_mask)
                    else:
                        mask_hint_data = predicted_mask
            else:
                mask_hint_data = None

            # === ENCODE POINTER HINT (into edge features) ===
            ptr_hint_node_fts, ptr_hint_edge_fts, ptr_hint_graph_fts, ptr_hint_adj = \
                self._encode_hint(ptr_hint_data, 'pointer', nb_nodes, device,
                                  batch_size=batch_size, hint_name='pi_h')

            # === ENCODE MASK HINT (into node features) ===
            if has_mask_hints:
                mask_hint_node_fts, mask_hint_edge_fts, mask_hint_graph_fts, mask_hint_adj = \
                    self._encode_hint(mask_hint_data, 'mask', nb_nodes, device,
                                      batch_size=batch_size, hint_name='reach_h')
            else:
                mask_hint_node_fts = torch.zeros_like(ptr_hint_node_fts)
                mask_hint_edge_fts = torch.zeros_like(ptr_hint_edge_fts)
                mask_hint_graph_fts = torch.zeros_like(ptr_hint_graph_fts)
                mask_hint_adj = torch.zeros_like(ptr_hint_adj)

            # Combine input features with BOTH hint features
            combined_node_fts = node_fts + ptr_hint_node_fts + mask_hint_node_fts
            combined_edge_fts = edge_fts + ptr_hint_edge_fts + mask_hint_edge_fts
            combined_graph_fts = graph_fts + ptr_hint_graph_fts + mask_hint_graph_fts
            combined_adj = ((adj_mat + ptr_hint_adj + mask_hint_adj) > 0).float()

            # === PROCESSOR STEP ===
            nxt_hidden = hidden
            for _ in range(self.nb_msg_passing_steps):
                if self.processor_type == 'hypare':
                    nxt_hidden, gate_values, gru_state = self.processor(
                        node_fts=combined_node_fts,
                        edge_fts=combined_edge_fts,
                        graph_fts=combined_graph_fts,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden,
                        gru_state=gru_state
                    )
                elif self.processor_type == 'mpnn_gru':
                    nxt_hidden, gru_state = self.processor(
                        node_fts=combined_node_fts,
                        edge_fts=combined_edge_fts,
                        graph_fts=combined_graph_fts,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden,
                        gru_state=gru_state
                    )
                else:
                    nxt_hidden = self.processor(
                        node_fts=combined_node_fts,
                        edge_fts=combined_edge_fts,
                        graph_fts=combined_graph_fts,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden
                    )

            # === DROPOUT (matching original CLRS) ===
            if self.training and self.dropout_prob > 0.0:
                nxt_hidden = F.dropout(nxt_hidden, p=self.dropout_prob, training=True)

            # === DECODE ===
            h_t = torch.cat([combined_node_fts, hidden, nxt_hidden], dim=-1)

            # All compared processors use the same full-adjacency MPNN contract;
            # decoder masking would otherwise confound the architecture ablation.
            use_inf_bias = False

            # Decode pointer hint (pi_h)
            if use_inf_bias:
                ptr_hint_pred = self.decoder.decode_hint(
                    h_t, combined_edge_fts, combined_adj)
            else:
                ptr_hint_pred = self.decoder.decode_hint(
                    h_t, combined_edge_fts, None)

            pointer_hint_preds.append(ptr_hint_pred)

            # Decode mask hint (reach_h) if present
            if has_mask_hints:
                mask_hint_pred = self.decoder.decode_mask_hint(h_t)
                mask_hint_preds.append(mask_hint_pred)

            # Decode output (always pointer for BFS). Keep each sample's
            # prediction from its own final valid execution step.
            if use_inf_bias:
                step_output = self.decoder.decode_output(
                    h_t, combined_edge_fts, combined_adj)
            else:
                step_output = self.decoder.decode_output(
                    h_t, combined_edge_fts, None)

            if lengths is None:
                output_pred = step_output
                hidden = nxt_hidden
            else:
                lengths_device = lengths.to(device)
                is_final = step == (lengths_device - 2).clamp(min=0)
                if output_pred is None:
                    output_pred = step_output
                output_pred = torch.where(
                    is_final[:, None, None], step_output, output_pred)
                active = step < (lengths_device - 1).clamp(min=1)
                hidden = torch.where(active[:, None, None], nxt_hidden, hidden)

        if output_pred is None:
            raise RuntimeError('No execution step was run; cannot decode output.')

        result = {
            'hint_preds': pointer_hint_preds,
            'output_pred': output_pred,
        }
        if has_mask_hints:
            result['mask_hint_preds'] = mask_hint_preds
        if gate_values is not None:
            result['gate_values'] = gate_values

        return result
