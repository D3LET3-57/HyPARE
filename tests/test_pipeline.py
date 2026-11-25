import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clrs_native_adapter import CLRSNativeAdapter
from src.models.exact_clrs import ExactCLRSBaseline
from src.models.exact_clrs.decoder import ExactCLRSPointerDecoder
from src.models.exact_clrs.mpnn import ExactCLRSMpnn
from src.training.trainer import Trainer


class RecoveryRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = CLRSNativeAdapter('bfs', length=8, batch_size=4,
                                        seed=7, num_samples=32)
        cls.batch = cls.adapter.get_batch()

    def make_model(self, architecture='hypare'):
        batch = self.batch
        return ExactCLRSBaseline(
            node_feat_dim=batch['node_features'].shape[-1],
            edge_feat_dim=batch['edge_features'].shape[-1],
            graph_feat_dim=batch['graph_features'].shape[-1],
            hidden_dim=32,
            processor_type=architecture,
        )

    def test_all_parameters_exist_before_optimizer(self):
        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters())
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(any('encoder.encoders.pi_h_ptr' in n
                            for n, _ in model.named_parameters()))
        self.assertFalse([n for n, p in model.named_parameters() if id(p) not in optimized])

    def test_padding_masks_follow_lengths(self):
        masks = self.batch['pointer_masks'][:, :, 0]
        expected = (torch.arange(masks.shape[1])[None, :]
                    < self.batch['lengths'][:, None]).float()
        torch.testing.assert_close(masks, expected)

    def test_forward_and_backward_for_all_architectures(self):
        batch = self.batch
        for architecture in ('mpnn', 'mpnn_gru', 'hypare'):
            with self.subTest(architecture=architecture):
                model = self.make_model(architecture)
                predictions = model(
                    node_features=batch['node_features'],
                    edge_features=batch['edge_features'],
                    graph_features=batch['graph_features'],
                    adjacency=batch['adjacency'],
                    pointer_targets=batch['pointer_targets'],
                    pointer_masks=batch['pointer_masks'],
                    mask_targets=batch['mask_targets'],
                    mask_hint_masks=batch['mask_hint_masks'],
                    lengths=batch['lengths'],
                    teacher_forcing=True,
                )
                trainer = Trainer(model, hint_loss_weight=1.0)
                loss, _, _ = trainer.compute_loss(
                    predictions, batch['pointer_targets'], batch['output_targets'],
                    batch['pointer_masks'], batch['node_features'].shape[1],
                    batch['mask_targets'], batch['mask_hint_masks'])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.net.encoder.encoders['pi_h_ptr'].weight.grad)

    def test_hypare_recurrent_state_is_node_specific(self):
        model = self.make_model('hypare')
        self.assertEqual(model.net.processor.gru.input_size, 32)
        self.assertFalse(hasattr(model.net.processor, 'gru_state'))
        self.assertFalse(hasattr(model.net.processor, 'gru_input'))

    def test_parameter_count_matches_official_bfs_mpnn(self):
        batch = self.batch
        model = ExactCLRSBaseline(
            node_feat_dim=batch['node_features'].shape[-1],
            edge_feat_dim=batch['edge_features'].shape[-1],
            graph_feat_dim=batch['graph_features'].shape[-1],
            hidden_dim=128,
            processor_type='mpnn',
            use_ln=True,
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 413827)
        self.assertIsNotNone(model.net.processor.layer_norm)

    def test_dynamic_num_steps_follow_hint_targets(self):
        batch = self.batch
        max_steps = min(4, batch['pointer_targets'].size(1))
        self.assertGreaterEqual(max_steps, 2)
        model = self.make_model('mpnn')
        predictions = model(
            node_features=batch['node_features'],
            edge_features=batch['edge_features'],
            graph_features=batch['graph_features'],
            adjacency=batch['adjacency'],
            pointer_targets=batch['pointer_targets'][:, :max_steps],
            hint_targets=batch['pointer_targets'][:, :max_steps],
            pointer_masks=batch['hint_masks'][:, :max_steps],
            hint_masks=batch['hint_masks'][:, :max_steps],
            mask_targets=batch['mask_targets'][:, :max_steps],
            mask_hint_masks=batch['mask_hint_masks'][:, :max_steps],
            lengths=batch['lengths'],
            teacher_forcing=False,
        )
        self.assertEqual(len(predictions['hint_preds']), max_steps - 1)

    def test_pointer_decoder_uses_receiver_key(self):
        decoder = ExactCLRSPointerDecoder(hidden_dim=2, decoder_input_dim=2)
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.zero_()
            decoder.p_2.weight[0, 0] = 1.0
            decoder.p_final.weight[0, 0] = 1.0

        h_t = torch.zeros(1, 2, 2)
        h_t[0, :, 0] = torch.tensor([0.0, 1.0])
        edge_fts = torch.zeros(1, 2, 2, 2)
        scores = decoder(h_t, edge_fts, None)
        self.assertAlmostEqual(scores[0, 0, 1].item(), 1.0)
        self.assertAlmostEqual(scores[0, 1, 1].item(), 1.0)

    def test_mpnn_messages_use_official_axes(self):
        processor = ExactCLRSMpnn(
            hidden_dim=2,
            msg_mlp_sizes=None,
            use_full_adjacency=True,
            use_ln=False,
        )
        with torch.no_grad():
            for parameter in processor.parameters():
                parameter.zero_()
            processor.msg_linear_1.weight[0, 0] = 1.0
            processor.update_linear_2.weight[0, 0] = 1.0
            processor.update_linear_2.weight[1, 1] = 1.0

        node_fts = torch.zeros(1, 2, 2)
        node_fts[0, :, 0] = torch.tensor([0.0, 1.0])
        hidden = torch.zeros(1, 2, 2)
        edge_fts = torch.zeros(1, 2, 2, 2)
        graph_fts = torch.zeros(1, 2)
        adj_mat = torch.ones(1, 2, 2)
        output = processor(node_fts, edge_fts, graph_fts, adj_mat, hidden)
        torch.testing.assert_close(
            output, torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]))

    def test_random_pos_is_sorted_and_changes_positions(self):
        fixed_adapter = CLRSNativeAdapter(
            'bfs', length=8, batch_size=4, seed=11, num_samples=16,
            randomize_pos=False)
        random_adapter = CLRSNativeAdapter(
            'bfs', length=8, batch_size=4, seed=11, num_samples=16,
            randomize_pos=True)
        fixed = fixed_adapter.get_batch(4)['node_features'][..., 0]
        randomized = random_adapter.get_batch(4)['node_features'][..., 0]
        self.assertTrue((randomized[:, 1:] >= randomized[:, :-1]).all())
        self.assertFalse(torch.allclose(fixed, randomized))


if __name__ == '__main__':
    unittest.main()
