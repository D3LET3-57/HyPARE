import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clrs_spec_adapter import CLRSSpecAdapter
from src.models.exact_clrs.spec_model import SpecCLRSNet
from src.training.spec_trainer import compute_losses, compute_metrics


class SpecPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = CLRSSpecAdapter(
            'dfs', length=8, batch_size=2, seed=7, num_samples=16)
        cls.batch = cls.adapter.get_batch()

    def make_model(self, architecture):
        return SpecCLRSNet(
            self.batch['spec'],
            self.batch['categorical_dims'],
            hidden_dim=16,
            nb_msg_passing_steps=1,
            processor_type=architecture,
        )

    def test_dfs_hints_are_named_and_typed(self):
        hints = self.batch['hints']
        self.assertEqual(set(hints), {
            'pi_h', 'color', 'd', 'f', 's_prev',
            's', 'u', 'v', 's_last', 'time'})
        self.assertEqual(hints['pi_h']['target'].shape,
                         (2, self.batch['max_steps'], 8))
        self.assertEqual(hints['s']['target'].shape,
                         (2, self.batch['max_steps']))
        self.assertEqual(hints['time']['target'].shape,
                         (2, self.batch['max_steps']))
        self.assertEqual(self.batch['categorical_dims']['color'], 3)
        self.assertEqual(self.batch['outputs']['pi']['target'].shape, (2, 8))

    def test_forward_loss_backward_for_all_architectures(self):
        for architecture in ('mpnn', 'mpnn_gru', 'hypare'):
            with self.subTest(architecture=architecture):
                model = self.make_model(architecture)
                output = model(
                    self.batch['inputs'],
                    self.batch['hints'],
                    lengths=self.batch['lengths'],
                    teacher_forcing_ratio=0.5,
                )
                self.assertEqual(
                    output['output_preds']['pi'].shape, (2, 8, 8))
                losses = compute_losses(output, self.batch)
                self.assertTrue(torch.isfinite(losses['total']))
                losses['total'].backward()
                gradients = [p.grad for p in model.parameters()
                             if p.grad is not None]
                self.assertTrue(gradients)

    def test_metrics_are_finite(self):
        model = self.make_model('mpnn')
        output = model(
            self.batch['inputs'],
            self.batch['hints'],
            lengths=self.batch['lengths'],
            teacher_forcing_ratio=0.0,
        )
        metrics = compute_metrics(output, self.batch)
        self.assertGreaterEqual(metrics['output_accuracy'], 0.0)
        self.assertLessEqual(metrics['output_accuracy'], 1.0)
        self.assertGreaterEqual(metrics['hint_accuracy'], 0.0)
        self.assertLessEqual(metrics['hint_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
