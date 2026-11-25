#!/usr/bin/env python3
"""Reproducible BFS comparison for MPNN, MPNN+GRU, and HyPARE."""

import argparse
import json

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clrs._src import samplers
from src.data.clrs_native_adapter import CLRSNativeAdapter
from src.models.exact_clrs import ExactCLRSBaseline
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architectures', nargs='+',
                        choices=['mpnn', 'mpnn_gru', 'hypare'],
                        default=['mpnn', 'mpnn_gru', 'hypare'])
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--teacher-forcing', type=float, default=0.5)
    parser.add_argument('--use-ln', action=argparse.BooleanOptionalAction,
                        default=True, help='Apply official CLRS LayerNorm after MPNN update')
    parser.add_argument('--random-pos', action=argparse.BooleanOptionalAction,
                        default=True, help='Randomize ordered pos inputs for train/val')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'v4_bfs.json')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints_v4')
    parser.add_argument('--smoke', action='store_true', help='Run 5 steps on tiny batches')
    parser.add_argument('--resume', action='store_true',
                        help='Resume each run from checkpoints/latest_* when present')
    parser.add_argument('--lr-schedule', choices=['none', 'cosine'], default='none',
                        help='Learning rate schedule')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Adam weight decay (L2 regularization)')
    parser.add_argument('--gate-init', type=float, default=-2.0,
                        help='Initial bias of HyPARE gate pre-sigmoid layer')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def forward(model, batch, teacher_forcing):
    return model(
        node_features=batch['node_features'],
        edge_features=batch['edge_features'],
        graph_features=batch['graph_features'],
        adjacency=batch['adjacency'],
        hint_targets=batch['hint_targets'],
        hint_masks=batch['hint_masks'],
        pointer_targets=batch['pointer_targets'],
        pointer_masks=batch['pointer_masks'],
        mask_targets=batch['mask_targets'],
        mask_hint_masks=batch['mask_hint_masks'],
        lengths=batch['lengths'],
        teacher_forcing=teacher_forcing,
    )


def accuracy(model, adapter, device, batch_size):
    model.eval()
    batch = move_batch(adapter.get_batch(batch_size=batch_size), device)
    with torch.no_grad():
        predictions = forward(model, batch, teacher_forcing=False)
    predicted = predictions['output_pred'].argmax(dim=-1)
    return (predicted == batch['output_targets']).float().mean().item()


def train_one(architecture, seed, args):
    seed_everything(seed)
    split = samplers.CLRS30
    train = CLRSNativeAdapter('bfs', 'train', 16, args.batch_size, 1,
                              split['train']['num_samples'],
                              randomize_pos=args.random_pos)
    val = CLRSNativeAdapter('bfs', 'val', 16, args.batch_size, 2,
                            split['val']['num_samples'],
                            randomize_pos=args.random_pos)
    test = CLRSNativeAdapter('bfs', 'test', 64, args.batch_size, 3,
                             split['test']['num_samples'],
                             randomize_pos=False)
    sample = train.get_batch()
    model = ExactCLRSBaseline(
        node_feat_dim=sample['node_features'].shape[-1],
        edge_feat_dim=sample['edge_features'].shape[-1],
        graph_feat_dim=sample['graph_features'].shape[-1],
        hidden_dim=args.hidden_dim,
        teacher_forcing_ratio=args.teacher_forcing,
        processor_type=architecture,
        gate_init=args.gate_init,
        use_ln=args.use_ln,
    ).to(args.device)
    trainer = Trainer(model, hint_loss_weight=1.0, hint_type='pointer',
                      output_type='pointer', device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

    # Optional learning rate scheduler.
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.01)

    # This assertion prevents the lazy-encoder regression from silently returning.
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    missing = [name for name, parameter in model.named_parameters()
               if id(parameter) not in optimized]
    if missing:
        raise RuntimeError(f'Parameters missing from optimizer: {missing}')

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f'latest_bfs_{architecture}_seed{seed}.pt'
    best_val = -1.0
    best_state = None
    start_step = 0
    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        best_val = state['best_val']
        best_state = state['best_state']
        start_step = state['step']
        if scheduler is not None and 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        print(f'Resumed {architecture} seed={seed} from step {start_step}', flush=True)

    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        batch = move_batch(train.get_batch(), args.device)
        predictions = forward(model, batch, teacher_forcing=True)
        loss, hint_loss, output_loss = trainer.compute_loss(
            predictions, batch['hint_targets'], batch['output_targets'],
            batch['hint_masks'], batch['node_features'].shape[1],
            batch['mask_targets'], batch['mask_hint_masks'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_acc = accuracy(model, val, args.device, 32)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'architecture': architecture,
                'seed': seed,
                'step': step,
                'best_val': best_val,
                'best_state': best_state,
            }, latest_path)
            print(f'{architecture:9s} seed={seed} step={step:5d} '
                  f'loss={loss.item():.4f} hint={hint_loss.item():.4f} '
                  f'output={output_loss.item():.4f} val={val_acc:.4f}', flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = accuracy(model, test, args.device, 32)
    torch.save({'model': model.state_dict(), 'architecture': architecture,
                'seed': seed, 'config': vars(args)},
               checkpoint_dir / f'bfs_{architecture}_seed{seed}.pt')
    return {'val': best_val, 'test_ood': test_acc,
            'parameters': sum(p.numel() for p in model.parameters()),
            'seconds': time.time() - started}


def main():
    args = parse_args()
    if args.smoke:
        args.steps = 5
        args.eval_every = 5
        args.batch_size = 4
        args.hidden_dim = 32
        args.seeds = [0]
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; pass --device cpu')

    results = {'config': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, 'runs': {}}
    for architecture in args.architectures:
        results['runs'][architecture] = {}
        for seed in args.seeds:
            results['runs'][architecture][str(seed)] = train_one(
                architecture, seed, args)

    for architecture, runs in results['runs'].items():
        scores = [run['test_ood'] for run in runs.values()]
        results.setdefault('summary', {})[architecture] = {
            'test_mean': float(np.mean(scores)),
            'test_std': float(np.std(scores)),
            'test_scores': scores,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results['summary'], indent=2))
    print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()
