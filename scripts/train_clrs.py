#!/usr/bin/env python3
"""Train the spec-driven model on any single CLRS algorithm (DFS first)."""

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
from src.data.clrs_spec_adapter import CLRSSpecAdapter
from src.models.exact_clrs.spec_model import SpecCLRSNet
from src.training.spec_trainer import train_spec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='dfs')
    parser.add_argument('--architectures', nargs='+',
                        choices=['mpnn', 'mpnn_gru', 'hypare'],
                        default=['mpnn'])
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--teacher-forcing', type=float, default=0.5)
    parser.add_argument('--use-ln', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--random-pos', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--test-every', type=int, default=0)
    parser.add_argument('--train-length', type=int, default=16)
    parser.add_argument('--val-length', type=int, default=16)
    parser.add_argument('--test-length', type=int, default=64)
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--val-samples', type=int, default=32)
    parser.add_argument('--test-samples', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'v4_dfs.json')
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=ROOT / 'checkpoints_v4')
    parser.add_argument('--log-file', type=Path, default=None)
    parser.add_argument('--gate-init', type=float, default=-2.0)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(architecture, seed, args):
    seed_everything(seed)
    split = samplers.CLRS30
    train = CLRSSpecAdapter(
        args.algorithm, 'train', args.train_length, args.batch_size,
        split['train']['seed'], args.train_samples,
        randomize_pos=args.random_pos)
    val = CLRSSpecAdapter(
        args.algorithm, 'val', args.val_length, args.batch_size,
        split['val']['seed'], args.val_samples,
        randomize_pos=args.random_pos)
    test = CLRSSpecAdapter(
        args.algorithm, 'test', args.test_length, args.batch_size,
        split['test']['seed'], args.test_samples,
        randomize_pos=False)

    sample = train.get_batch()
    model = SpecCLRSNet(
        sample['spec'],
        sample['categorical_dims'],
        hidden_dim=args.hidden_dim,
        nb_msg_passing_steps=1,
        processor_type=architecture,
        gate_init=args.gate_init,
        use_ln=args.use_ln,
    ).to(args.device)

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f'{args.algorithm}_{architecture}_seed{seed}.pt')
    log_path = args.log_file
    if log_path is None:
        log_dir = ROOT / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            f'v4_{args.algorithm}_{architecture}_seed{seed}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = train_spec(
        model, train, val, test,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing,
        eval_every=args.eval_every,
        test_every=args.test_every,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )
    torch.save({
        'model': model.state_dict(),
        'architecture': architecture,
        'seed': seed,
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'result': result['test'],
    }, checkpoint_path)
    return {
        'best_valid': result['best_valid'],
        'valid': result['valid'],
        'test': result['test'],
        'test_ood': result['test']['output_accuracy'],
        'test_hint': result['test']['hint_accuracy'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'seconds': time.time() - started,
        'history': result['history'],
        'checkpoint': str(checkpoint_path),
        'log_file': str(log_path),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.steps = 5
        args.eval_every = 2
        args.batch_size = 4
        args.hidden_dim = 32
        args.train_length = 8
        args.val_length = 8
        args.test_length = 8
        args.train_samples = 16
        args.val_samples = 8
        args.test_samples = 8
        args.seeds = [0]
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; pass --device cpu')

    results = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'runs': {},
    }
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
