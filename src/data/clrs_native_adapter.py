"""
Native CLRS Data Adapter.

Uses the original DeepMind CLRS library (JAX) as a data generation API,
converting outputs to PyTorch tensors for your PyTorch models.

This ensures:
1. Data is generated exactly as the original CLRS benchmark intends
2. No discrepancies in sampling, padding, or hint structure
3. Your PyTorch models receive correctly formatted data

Usage:
    adapter = CLRSNativeAdapter(algorithm='bfs', split='train', length=16, batch_size=32)
    batch = adapter.get_batch()
    # batch is a dict with PyTorch tensors:
    #   node_features: [batch, num_nodes, node_feat_dim]
    #   edge_features: [batch, num_nodes, num_nodes, edge_feat_dim]
    #   graph_features: [batch, graph_feat_dim]
    #   adjacency: [batch, num_nodes, num_nodes]
    #   hint_targets: [batch, max_steps, num_nodes, ...]
    #   output_targets: [batch, num_nodes, ...]
    #   hint_masks: [batch, max_steps, num_nodes]
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any

# Import the original CLRS library
import clrs
from clrs._src import samplers
from clrs._src import specs


class CLRSNativeAdapter:
    """Adapter that uses original CLRS samplers to generate data for PyTorch models.

    This wraps the official DeepMind CLRS sampling pipeline and converts
    the outputs to PyTorch tensors compatible with your BaselineModel.
    """

    def __init__(self, algorithm: str, split: str = 'train',
                 length: int = 16, batch_size: int = 32,
                 seed: int = 42, num_samples: int = -1,
                 randomize_pos: bool = True):
        """Initialize the adapter.

        Args:
            algorithm: Algorithm name (e.g., 'bfs', 'dfs', 'dijkstra')
            split: 'train', 'val', or 'test'
            length: Graph size (number of nodes)
            batch_size: Batch size for training
            seed: Random seed
            num_samples: Number of samples (-1 for on-the-fly generation)
            randomize_pos: Randomize the ordered `pos` input like official CLRS
                training and validation. Disable for test.
        """
        self.algorithm = algorithm
        self.split = split
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.randomize_pos = randomize_pos
        self._pos_rng = np.random.RandomState(seed)

        # Build the sampler using original CLRS
        self.sampler, self.spec = samplers.build_sampler(
            name=algorithm,
            num_samples=num_samples,
            length=length,
            seed=seed,
            track_max_steps=True
        )

        # Store spec info for feature extraction
        self._spec = specs.SPECS[algorithm]
        self._feature_dims = self._compute_feature_dims()

    def _compute_feature_dims(self) -> Dict[str, Any]:
        """Compute feature dimensions from the spec."""
        node_feat_dim = 1  # Default: scalar per node
        edge_feat_dim = 1  # Default: scalar per edge
        graph_feat_dim = 0  # No synthetic graph feature when the spec has none

        # Analyze spec to determine feature dimensions
        for name, (stage, loc, typ) in self._spec.items():
            if stage == specs.Stage.INPUT:
                if loc == specs.Location.NODE:
                    node_feat_dim = max(node_feat_dim, 1)
                elif loc == specs.Location.EDGE:
                    edge_feat_dim = max(edge_feat_dim, 1)
                elif loc == specs.Location.GRAPH:
                    graph_feat_dim = max(graph_feat_dim, 1)

        # Determine hint/output types
        hint_type = 'pointer'
        output_type = 'pointer'
        num_hint_classes = 2
        num_output_classes = 2

        for name, (stage, loc, typ) in self._spec.items():
            if stage == specs.Stage.HINT:
                if typ == specs.Type.POINTER:
                    hint_type = 'pointer'
                elif typ in [specs.Type.MASK, specs.Type.MASK_ONE]:
                    hint_type = 'mask'
                elif typ == specs.Type.CATEGORICAL:
                    hint_type = 'categorical'
            if stage == specs.Stage.OUTPUT:
                if typ == specs.Type.POINTER:
                    output_type = 'pointer'
                elif typ in [specs.Type.MASK, specs.Type.MASK_ONE]:
                    output_type = 'mask'
                elif typ == specs.Type.CATEGORICAL:
                    output_type = 'categorical'

        return {
            'node_feat_dim': node_feat_dim,
            'edge_feat_dim': edge_feat_dim,
            'graph_feat_dim': graph_feat_dim,
            'hint_type': hint_type,
            'output_type': output_type,
            'num_hint_classes': num_hint_classes,
            'num_output_classes': num_output_classes,
            'max_steps': 0,  # Will be set dynamically
        }

    def get_feature_dims(self) -> Dict[str, Any]:
        """Get feature dimensions."""
        return self._feature_dims.copy()

    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get a batch of data as PyTorch tensors.

        Args:
            batch_size: Override batch size (uses self.batch_size if None)

        Returns:
            dict with keys:
                node_features: [B, N, node_feat_dim]
                edge_features: [B, N, N, edge_feat_dim]
                graph_features: [B, graph_feat_dim]
                adjacency: [B, N, N]
                hint_targets: [B, max_steps, num_nodes] (pointer) or [B, max_steps, num_nodes, classes]
                output_targets: [B, num_nodes] (pointer) or [B, num_nodes, classes]
                hint_masks: [B, max_steps, num_nodes]
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Get batch from original CLRS sampler
        feedback = self.sampler.next(batch_size=batch_size)
        if self.randomize_pos:
            feedback = self._randomize_pos(feedback)

        # Convert to PyTorch tensors
        return self._feedback_to_tensors(feedback)

    def _randomize_pos(self, feedback):
        """Replace ordered `pos` values with sorted random values per split.

        Mirrors clrs._src.samplers.process_random_pos. BFS and DFS have a
        single position segment; the split handling is retained for parity.
        """
        pos_inputs = [x for x in feedback.features.inputs if x.name == 'pos']
        if not pos_inputs:
            return feedback

        pos = pos_inputs[0]
        batch_size, num_nodes = pos.data.shape
        unsorted = self._pos_rng.uniform(size=(batch_size, num_nodes))
        new_pos = []
        for i in range(batch_size):
            split, = np.where(pos.data[i] == 0)
            split = np.concatenate([split, [num_nodes]])
            new_pos.append(np.concatenate([
                np.sort(unsorted[i, split[j]:split[j + 1]])
                for j in range(len(split) - 1)
            ]))
        pos.data = np.array(new_pos)
        inputs = [(pos if x.name == 'pos' else x)
                  for x in feedback.features.inputs]
        features = feedback.features._replace(inputs=inputs)
        return feedback._replace(features=features)

    def _feedback_to_tensors(self, feedback) -> Dict[str, torch.Tensor]:
        """Convert CLRS Feedback to PyTorch tensors.

        Args:
            feedback: clrs._src.samplers.Feedback object

        Returns:
            dict of PyTorch tensors
        """
        batch_size = None
        num_nodes = None

        # === EXTRACT INPUT FEATURES ===
        node_features_list = []
        edge_features_list = []
        adjacency = None

        for inp in feedback.features.inputs:
            data = torch.from_numpy(np.array(inp.data)).float()

            if batch_size is None:
                batch_size = data.shape[0]
            if num_nodes is None and inp.location in [specs.Location.NODE, specs.Location.EDGE]:
                num_nodes = data.shape[1]

            if inp.location == specs.Location.NODE:
                # Node input: [B, N] -> [B, N, 1]
                if data.dim() == 2:
                    data = data.unsqueeze(-1)
                node_features_list.append(data)

            elif inp.location == specs.Location.EDGE:
                # Edge input: [B, N, N] -> [B, N, N, 1]
                if data.dim() == 3:
                    data = data.unsqueeze(-1)
                # CLRS uses adjacency both for connectivity and as an encoded edge input.
                if inp.name == 'adj' or 'adjacency' in inp.name.lower():
                    if adjacency is None:
                        raw_adj = data.squeeze(-1)
                        adjacency = ((raw_adj + raw_adj.transpose(1, 2)) > 0.0).float()
                edge_features_list.append(data)

            elif inp.location == specs.Location.GRAPH:
                # Graph input: [B] -> [B, 1]
                if data.dim() == 1:
                    data = data.unsqueeze(-1)
                if 'graph_features_list' not in locals():
                    graph_features_list = []
                graph_features_list.append(data)

        # Combine node features
        if node_features_list:
            node_features = torch.cat(node_features_list, dim=-1)
        else:
            node_features = torch.zeros(batch_size, num_nodes, 1)

        # Combine edge features
        if edge_features_list:
            edge_features = torch.cat(edge_features_list, dim=-1)
        else:
            edge_features = torch.zeros(batch_size, num_nodes, num_nodes, 1)

        # Default adjacency (fully connected if not specified)
        if adjacency is None:
            adjacency = torch.ones(batch_size, num_nodes, num_nodes)

        # Do not invent a learned global bias when the algorithm has no graph input.
        if 'graph_features_list' in locals() and graph_features_list:
            graph_features = torch.cat(graph_features_list, dim=-1)
        else:
            graph_features = torch.zeros(batch_size, 0)

        # === EXTRACT HINT TARGETS ===
        # Match original CLRS: extract ALL hint types separately
        # NODE POINTER hints (pi_h) -> pointer_targets
        # NODE MASK hints (reach_h) -> mask_targets
        max_steps = 0
        lengths = torch.from_numpy(np.asarray(feedback.features.lengths)).long()

        # Find all hint types
        pointer_hints = [h for h in feedback.features.hints
                        if h.type_ == specs.Type.POINTER and h.location == specs.Location.NODE]
        mask_hints = [h for h in feedback.features.hints
                     if h.type_ in [specs.Type.MASK, specs.Type.MASK_ONE]
                     and h.location == specs.Location.NODE]

        # Extract pointer hint targets (for pi_h)
        pointer_targets = torch.zeros(batch_size, max_steps, num_nodes, dtype=torch.long)
        pointer_masks = torch.zeros(batch_size, max_steps, num_nodes)

        if pointer_hints:
            hint = pointer_hints[0]
            data = torch.from_numpy(np.array(hint.data)).long()
            if data.dim() == 3:  # [max_steps, batch, nodes]
                data = data.transpose(0, 1)  # [batch, max_steps, nodes]
                max_steps = max(max_steps, data.shape[1])
                pointer_targets = torch.zeros(batch_size, data.shape[1], num_nodes, dtype=torch.long)
                pointer_masks = torch.zeros(batch_size, data.shape[1], num_nodes)
                pointer_targets[:, :data.shape[1], :data.shape[2]] = data
                valid_steps = torch.arange(data.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
                pointer_masks = valid_steps.unsqueeze(-1).expand(-1, -1, num_nodes).float()
            elif data.dim() == 2:  # [batch, nodes] - one-hot pointer
                indices = data.argmax(dim=-1)  # [batch, nodes]
                max_steps = max(max_steps, 1)
                pointer_targets = torch.zeros(batch_size, 1, num_nodes, dtype=torch.long)
                pointer_masks = torch.zeros(batch_size, 1, num_nodes)
                pointer_targets[:, 0, :] = indices
                pointer_masks[:, 0, :] = 1.0

        # Extract mask hint targets (for reach_h)
        mask_targets = torch.zeros(batch_size, max_steps, num_nodes)
        mask_hint_masks = torch.zeros(batch_size, max_steps, num_nodes)

        if mask_hints:
            hint = mask_hints[0]
            data = torch.from_numpy(np.array(hint.data)).float()
            if data.dim() == 3:  # [max_steps, batch, nodes]
                data = data.transpose(0, 1)  # [batch, max_steps, nodes]
                max_steps = max(max_steps, data.shape[1])
                mask_targets = torch.zeros(batch_size, data.shape[1], num_nodes)
                mask_hint_masks = torch.zeros(batch_size, data.shape[1], num_nodes)
                mask_targets[:, :data.shape[1], :data.shape[2]] = data
                valid_steps = torch.arange(data.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
                mask_hint_masks = valid_steps.unsqueeze(-1).expand(-1, -1, num_nodes).float()
            elif data.dim() == 2:  # [batch, nodes]
                max_steps = max(max_steps, 1)
                mask_targets = torch.zeros(batch_size, 1, num_nodes)
                mask_hint_masks = torch.zeros(batch_size, 1, num_nodes)
                mask_targets[:, 0, :] = data
                mask_hint_masks[:, 0, :] = 1.0

        # For backward compatibility, set hint_targets to pointer targets
        hint_targets = pointer_targets
        hint_masks = pointer_masks

        # === EXTRACT OUTPUT TARGETS ===
        output_targets = torch.zeros(batch_size, num_nodes)

        for out in feedback.outputs:
            if out.location == specs.Location.NODE:
                if out.type_ == specs.Type.POINTER:
                    # Pointer output: [batch, nodes, nodes] -> argmax -> [batch, nodes]
                    data = torch.from_numpy(np.array(out.data))
                    if data.dim() == 3:
                        output_targets = data.argmax(dim=-1).long()
                    elif data.dim() == 2:
                        output_targets = data.long()
                elif out.type_ in [specs.Type.MASK, specs.Type.MASK_ONE]:
                    data = torch.from_numpy(np.array(out.data)).float()
                    if data.dim() == 2:
                        output_targets = data
                break  # Use first node output as primary

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'graph_features': graph_features,
            'adjacency': adjacency,
            'hint_targets': hint_targets,
            'output_targets': output_targets,
            'hint_masks': hint_masks,
            'pointer_targets': pointer_targets,
            'pointer_masks': pointer_masks,
            'mask_targets': mask_targets,
            'mask_hint_masks': mask_hint_masks,
            'num_nodes': num_nodes,
            'max_steps': max_steps,
            'lengths': lengths,
        }


def get_native_feature_dims(algorithm: str, length: int = 16) -> Dict[str, Any]:
    """Get feature dimensions for an algorithm using native CLRS.

    Counts actual number of input features per location from the spec.
    """
    spec = specs.SPECS[algorithm]

    node_feat_dim = 0
    edge_feat_dim = 0
    graph_feat_dim = 0

    for name, (stage, loc, typ) in spec.items():
        if stage == specs.Stage.INPUT:
            if loc == specs.Location.NODE:
                node_feat_dim += 1
            elif loc == specs.Location.EDGE:
                edge_feat_dim += 1
            elif loc == specs.Location.GRAPH:
                graph_feat_dim += 1

    # Ensure minimum of 1
    node_feat_dim = max(node_feat_dim, 1)
    edge_feat_dim = max(edge_feat_dim, 1)
    graph_feat_dim = max(graph_feat_dim, 1)

    # Determine types
    hint_type = 'pointer'
    output_type = 'pointer'

    for name, (stage, loc, typ) in spec.items():
        if stage == specs.Stage.HINT:
            if typ == specs.Type.POINTER:
                hint_type = 'pointer'
            elif typ in [specs.Type.MASK, specs.Type.MASK_ONE]:
                hint_type = 'mask'
        if stage == specs.Stage.OUTPUT:
            if typ == specs.Type.POINTER:
                output_type = 'pointer'
            elif typ in [specs.Type.MASK, specs.Type.MASK_ONE]:
                output_type = 'mask'

    return {
        'node_feat_dim': node_feat_dim,
        'edge_feat_dim': edge_feat_dim,
        'graph_feat_dim': graph_feat_dim,
        'hint_type': hint_type,
        'output_type': output_type,
        'num_hint_classes': length,  # For pointer, num_classes = num_nodes
        'num_output_classes': length,
    }
