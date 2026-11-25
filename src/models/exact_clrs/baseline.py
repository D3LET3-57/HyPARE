"""
Exact CLRS Baseline Model — PyTorch wrapper for ExactCLRSNet.

This provides a clean interface matching our existing training pipeline
while using the exact CLRS implementation underneath.
"""

import torch
import torch.nn as nn
from .net import ExactCLRSNet


class ExactCLRSBaseline(nn.Module):
    """Exact CLRS Baseline model wrapper.

    Args:
        node_feat_dim: Dimension of node features (typically 1)
        edge_feat_dim: Dimension of edge features (typically 1)
        graph_feat_dim: Dimension of graph features (typically 1)
        hidden_dim: Dimension of hidden states
        hint_type: 'pointer', 'mask', 'mask_one', or 'categorical'
        output_type: 'pointer', 'mask', 'mask_one', or 'categorical'
        num_hint_classes: Number of classes for non-pointer hints
        num_output_classes: Number of classes for non-pointer outputs
        num_steps: Number of execution steps
        nb_msg_passing_steps: Number of message passing steps per hint step
        teacher_forcing_ratio: Probability of using ground-truth hints
    """

    def __init__(self, node_feat_dim: int = 1, edge_feat_dim: int = 1,
                 graph_feat_dim: int = 1, hidden_dim: int = 128,
                 hint_type: str = 'pointer', output_type: str = 'pointer',
                 num_hint_classes: int = 2, num_output_classes: int = 2,
                 num_steps: int | None = None, nb_msg_passing_steps: int = 1,
                 teacher_forcing_ratio: float = 0.0,
                 dropout_prob: float = 0.0,
                 processor_type: str = 'mpnn',
                 gate_init: float = -2.0,
                 use_ln: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.hint_type = hint_type
        self.output_type = output_type
        self.num_steps = num_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.processor_type = processor_type

        # Exact CLRS Net
        self.net = ExactCLRSNet(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            graph_feat_dim=graph_feat_dim,
            hidden_dim=hidden_dim,
            nb_msg_passing_steps=nb_msg_passing_steps,
            hint_type=hint_type,
            output_type=output_type,
            num_hint_classes=num_hint_classes,
            num_output_classes=num_output_classes,
            dropout_prob=dropout_prob,
            processor_type=processor_type,
            gate_init=gate_init,
            use_ln=use_ln,
        )

    def forward(self, node_features, edge_features=None, graph_features=None,
                adjacency=None, hint_targets=None, hint_masks=None,
                pointer_targets=None, pointer_masks=None,
                mask_targets=None, mask_hint_masks=None, lengths=None,
                num_steps=None, teacher_forcing=None):
        """Forward pass matching our existing training pipeline.

        Args:
            node_features: [batch, num_nodes, node_feat_dim]
            edge_features: [batch, num_nodes, num_nodes, edge_feat_dim]
            graph_features: [batch, graph_feat_dim]
            adjacency: [batch, num_nodes, num_nodes]
            hint_targets: [batch, max_steps, num_nodes] - ground-truth hints (pointer)
            hint_masks: [batch, max_steps, num_nodes] - hint validity masks
            pointer_targets: [batch, max_steps, num_nodes] - pointer hint targets
            pointer_masks: [batch, max_steps, num_nodes] - pointer hint masks
            mask_targets: [batch, max_steps, num_nodes] - mask hint targets (0/1)
            mask_hint_masks: [batch, max_steps, num_nodes] - mask hint validity masks
            num_steps: Number of execution steps
            teacher_forcing: Whether to use teacher forcing (bool)

        Returns:
            dict with 'hint_preds', 'output_pred', 'mask_hint_preds' (if mask hints)
        """
        if num_steps is None:
            num_steps = self.num_steps

        # Determine teacher forcing ratio
        if teacher_forcing is None:
            tf_ratio = self.teacher_forcing_ratio if self.training else 0.0
        elif teacher_forcing:
            tf_ratio = self.teacher_forcing_ratio
        else:
            tf_ratio = 0.0

        return self.net(
            node_features=node_features,
            edge_features=edge_features,
            graph_features=graph_features,
            adjacency=adjacency,
            hint_targets=hint_targets,
            hint_masks=hint_masks,
            pointer_targets=pointer_targets,
            pointer_masks=pointer_masks,
            mask_targets=mask_targets,
            mask_hint_masks=mask_hint_masks,
            lengths=lengths,
            num_steps=num_steps,
            teacher_forcing_ratio=tf_ratio,
            nb_nodes=node_features.size(1)
        )
