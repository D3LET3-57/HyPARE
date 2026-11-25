"""
Training loop for HyPARE models.

Implements:
- Training with hint supervision at each step
- Loss computation for hints and final outputs (size-invariant for pointers)
- Noisy teacher forcing (CLRS paper Section 4.1)
- Logging and checkpointing

Key change: For pointer-type hints/outputs, the loss is computed using
CrossEntropyLoss over a [batch*num_nodes, num_nodes] prediction matrix,
which is inherently size-invariant and works for OOD evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os


class Trainer:
    """Trainer for neural algorithmic reasoning models.

    Args:
        model: The model to train
        optimizer: Optimizer (if None, uses Adam)
        learning_rate: Learning rate
        hint_loss_weight: Weight for hint loss vs output loss
        device: Device to train on
        hint_type: 'pointer', 'mask', 'mask_one', or 'categorical'
        output_type: 'pointer', 'mask', 'mask_one', or 'categorical'
    """

    def __init__(self, model, optimizer=None, learning_rate=1e-3,
                 hint_loss_weight=1.0, device='cpu',
                 hint_type='pointer', output_type='pointer'):
        self.model = model.to(device)
        self.device = device
        self.hint_loss_weight = hint_loss_weight
        self.hint_type = hint_type
        self.output_type = output_type

        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        # Loss functions
        # ignore_index=-1 handles padding in variable-length hint sequences
        self.hint_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.output_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def compute_loss(self, predictions, hint_targets, output_targets,
                     hint_masks=None, num_nodes=None,
                     mask_targets=None, mask_hint_masks=None):
        """Compute combined hint + output loss.

        For pointer type:
            hint_pred: [batch, num_nodes, num_nodes] scores
            hint_target: [batch, num_nodes] node indices
            Loss = CrossEntropy(pred.reshape(-1, num_nodes), target.reshape(-1))

        For mask/categorical type:
            hint_pred: [batch, num_nodes, num_classes] logits
            hint_target: [batch, num_nodes] class indices
            Loss = CrossEntropy(pred.reshape(-1, num_classes), target.reshape(-1))

        For mask hints (reach_h in BFS):
            mask_pred: [batch, num_nodes] logits
            mask_target: [batch, num_nodes] binary values (0/1)
            Loss = BCEWithLogitsLoss(pred, target)

        Args:
            predictions: Dict from model forward pass
            hint_targets: [batch, max_steps, num_nodes] ground-truth hints (indices)
            output_targets: [batch, num_nodes] ground-truth outputs (indices)
            hint_masks: Optional mask for valid hint steps [batch, max_steps, num_nodes]
            num_nodes: Number of nodes (for pointer loss computation)
            mask_targets: [batch, max_steps, num_nodes] mask hint targets (0/1)
            mask_hint_masks: [batch, max_steps, num_nodes] mask for valid mask hint steps

        Returns:
            total_loss: Combined loss value
            hint_loss: Hint prediction loss
            output_loss: Output prediction loss
        """
        hint_preds = predictions['hint_preds']
        output_pred = predictions['output_pred']

        if num_nodes is None:
            num_nodes = hint_targets.size(-1)

        # Prediction produced while consuming hint t is supervised by hint t+1.
        # Reduce over all valid elements, matching CLRS variable-length masking.
        hint_loss_sum = output_pred.new_tensor(0.0)
        hint_count = output_pred.new_tensor(0.0)
        for step, hint_pred in enumerate(hint_preds):
            target_step = step + 1
            if target_step >= hint_targets.size(1):
                break
            target = hint_targets[:, target_step].to(self.device)
            loss_values = nn.functional.cross_entropy(
                hint_pred.reshape(-1, hint_pred.size(-1)),
                target.reshape(-1), ignore_index=-1, reduction='none'
            ).reshape_as(target)
            mask = (
                hint_masks[:, target_step].to(self.device)
                if hint_masks is not None else torch.ones_like(loss_values)
            )
            hint_loss_sum = hint_loss_sum + (loss_values * mask).sum()
            hint_count = hint_count + mask.sum()
        hint_loss = hint_loss_sum / hint_count.clamp(min=1.0)

        # Mask hint loss (reach_h in BFS) - matching original CLRS multi-hint processing
        mask_hint_loss = output_pred.new_tensor(0.0)
        if mask_targets is not None and 'mask_hint_preds' in predictions:
            mask_loss_sum = output_pred.new_tensor(0.0)
            mask_count = output_pred.new_tensor(0.0)
            for step, mask_pred in enumerate(predictions['mask_hint_preds']):
                target_step = step + 1
                if target_step >= mask_targets.size(1):
                    break
                mask_target = mask_targets[:, target_step].to(self.device)
                loss_values = nn.functional.binary_cross_entropy_with_logits(
                    mask_pred, mask_target, reduction='none')
                mask = (
                    mask_hint_masks[:, target_step].to(self.device)
                    if mask_hint_masks is not None else torch.ones_like(loss_values)
                )
                mask_loss_sum = mask_loss_sum + (loss_values * mask).sum()
                mask_count = mask_count + mask.sum()
            mask_hint_loss = mask_loss_sum / mask_count.clamp(min=1.0)

        # Output loss
        output_targets = output_targets.to(self.device)
        if self.output_type == 'pointer':
            # output_pred: [batch, num_nodes, num_nodes]
            # output_targets: [batch, num_nodes]
            output_loss = self.output_loss_fn(
                output_pred.reshape(-1, output_pred.size(-1)),
                output_targets.reshape(-1)
            )
        else:
            output_loss = self.output_loss_fn(
                output_pred.reshape(-1, output_pred.size(-1)),
                output_targets.reshape(-1)
            )

        # Total loss: output + pointer hint + mask hint (matching original CLRS)
        total_loss = output_loss + self.hint_loss_weight * hint_loss + self.hint_loss_weight * mask_hint_loss

        return total_loss, hint_loss, output_loss

    def compute_hint_accuracy(self, predictions, hint_targets, hint_masks=None):
        """Compute hint accuracy across all steps.

        Returns:
            accuracy: Average accuracy over valid positions
        """
        hint_preds = predictions['hint_preds']
        total_correct = 0
        total_valid = 0

        for step, hint_pred in enumerate(hint_preds):
            target_step = step + 1
            if target_step >= hint_targets.size(1):
                break

            target = hint_targets[:, target_step]
            pred_classes = hint_pred.argmax(dim=-1)
            correct = (pred_classes == target).float()

            if hint_masks is not None:
                mask = hint_masks[:, target_step]
                correct = correct * mask
                total_valid += mask.sum().item()

            total_correct += correct.sum().item()

        return total_correct / max(total_valid, 1)

    def compute_output_accuracy(self, predictions, output_targets):
        """Compute output accuracy.

        Returns:
            accuracy: Fraction of correct output predictions
        """
        output_pred = predictions['output_pred']
        pred_classes = output_pred.argmax(dim=-1)
        correct = (pred_classes == output_targets).float()
        return correct.mean().item()

    def train_epoch(self, dataloader):
        """Train for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Dict with epoch metrics
        """
        self.model.train()
        total_loss = 0
        total_hint_loss = 0
        total_output_loss = 0
        num_batches = 0

        start_time = time.time()

        for batch in dataloader:
            # Move batch to device
            node_features = batch['node_features'].to(self.device)
            edge_features = batch.get('edge_features')
            if edge_features is not None:
                edge_features = edge_features.to(self.device)
            graph_features = batch.get('graph_features')
            if graph_features is not None:
                graph_features = graph_features.to(self.device)
            adjacency = batch['adjacency'].to(self.device)
            hint_targets = batch['hint_targets'].to(self.device)
            output_targets = batch['output_targets'].to(self.device)
            hint_masks = batch.get('hint_masks')
            if hint_masks is not None:
                hint_masks = hint_masks.to(self.device)

            # Forward pass (with teacher forcing during training)
            self.optimizer.zero_grad()
            predictions = self.model(
                node_features=node_features,
                edge_features=edge_features,
                graph_features=graph_features,
                adjacency=adjacency,
                teacher_forcing=True,
                hint_targets=hint_targets,
                hint_masks=hint_masks
            )

            # Compute loss
            loss, hint_loss, output_loss = self.compute_loss(
                predictions, hint_targets, output_targets, hint_masks
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_hint_loss += hint_loss.item()
            total_output_loss += output_loss.item()
            num_batches += 1

        epoch_time = time.time() - start_time

        return {
            'loss': total_loss / max(num_batches, 1),
            'hint_loss': total_hint_loss / max(num_batches, 1),
            'output_loss': total_output_loss / max(num_batches, 1),
            'time': epoch_time
        }

    def train(self, train_loader, val_loader=None, num_epochs=100,
              save_dir=None, log_interval=10):
        """Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            log_interval: How often to log metrics

        Returns:
            Dict with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_hint_loss': [],
            'train_output_loss': []
        }

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_hint_loss'].append(train_metrics['hint_loss'])
            history['train_output_loss'].append(train_metrics['output_loss'])

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])

                # Save best model
                if save_dir and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'),
                                        epoch, val_metrics)

            # Log
            if epoch % log_interval == 0:
                msg = (f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                       f"hint_loss={train_metrics['hint_loss']:.4f}, "
                       f"output_loss={train_metrics['output_loss']:.4f}, "
                       f"time={train_metrics['time']:.2f}s")
                if val_loader:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}"
                print(msg)

        return history

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on a dataset.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_hint_loss = 0
        total_output_loss = 0
        num_batches = 0

        for batch in dataloader:
            node_features = batch['node_features'].to(self.device)
            edge_features = batch.get('edge_features')
            if edge_features is not None:
                edge_features = edge_features.to(self.device)
            graph_features = batch.get('graph_features')
            if graph_features is not None:
                graph_features = graph_features.to(self.device)
            adjacency = batch['adjacency'].to(self.device)
            hint_targets = batch['hint_targets'].to(self.device)
            output_targets = batch['output_targets'].to(self.device)
            hint_masks = batch.get('hint_masks')
            if hint_masks is not None:
                hint_masks = hint_masks.to(self.device)

            # No teacher forcing during evaluation
            predictions = self.model(
                node_features=node_features,
                edge_features=edge_features,
                graph_features=graph_features,
                adjacency=adjacency,
                teacher_forcing=False
            )

            loss, hint_loss, output_loss = self.compute_loss(
                predictions, hint_targets, output_targets, hint_masks
            )

            total_loss += loss.item()
            total_hint_loss += hint_loss.item()
            total_output_loss += output_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'hint_loss': total_hint_loss / max(num_batches, 1),
            'output_loss': total_output_loss / max(num_batches, 1)
        }

    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
