#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clrs._src import specs
from src.data.clrs_spec_adapter import CLRSSpecAdapter
from src.models.exact_clrs.spec_model import SpecCLRSNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='dfs')
    parser.add_argument('--architectures', nargs='+',
                        default=['mpnn', 'mpnn_gru', 'hypare'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[16, 24, 32, 48, 64])
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--extended-samples', type=int, default=256)
    parser.add_argument('--extended-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-seed', type=int, default=3)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=ROOT / 'checkpoints_v4')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'v4_dfs_diagnostics.json')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def checkpoint_path(args, arch, seed):
    return args.checkpoint_dir / f'{args.algorithm}_{arch}_seed{seed}.pt'


def load_fixed_batch(args, length, samples):
    adapter = CLRSSpecAdapter(
        args.algorithm, 'test', length, args.batch_size,
        args.test_seed, samples, randomize_pos=False)
    return adapter.get_full_batch()


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def _slice(value, start, end):
    return value[start:end] if torch.is_tensor(value) else value


def slice_batch(batch, start, end):
    out = {
        'inputs': {
            name: _slice(value, start, end)
            for name, value in batch['inputs'].items()
        },
        'hints': {
            name: {
                key: _slice(value, start, end)
                for key, value in hint.items()
            }
            for name, hint in batch['hints'].items()
        },
        'outputs': {
            name: {
                key: _slice(value, start, end)
                for key, value in output.items()
            }
            for name, output in batch['outputs'].items()
        },
        'lengths': batch['lengths'][start:end],
        'num_nodes': batch['num_nodes'],
        'max_steps': batch['max_steps'],
        'spec': batch['spec'],
        'categorical_dims': batch['categorical_dims'],
    }
    return out


def _accuracy_counts(prediction, target, mask, typ):
    if typ == specs.Type.POINTER:
        pred = prediction.argmax(dim=-1)
        target = target.long()
        correct = (pred == target).float() * mask.float()
        return correct, mask.float()
    if typ == specs.Type.MASK:
        pred = (prediction > 0).float()
        target = target.float()
        correct = (pred == target).float() * mask.float()
        return correct, mask.float()
    if typ in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL):
        pred = prediction.argmax(dim=-1)
        target = target.long()
        correct = (pred == target).float() * mask.float()
        return correct, mask.float()
    return None, None


def eval_chunk(model, batch):
    with torch.no_grad():
        output = model(
            batch['inputs'],
            batch['hints'],
            lengths=batch['lengths'],
            teacher_forcing_ratio=0.0,
        )

    result = {
        'n_graphs': int(batch['lengths'].numel()),
        'per_graph': {},
        'hints': {},
    }

    spec = batch['spec']
    for name, prediction in output['output_preds'].items():
        typ = spec[name][2]
        truth = batch['outputs'][name]
        correct, mask = _accuracy_counts(
            prediction, truth['target'], truth['mask'], typ)
        if correct is None:
            continue
        per_graph = correct.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        result['per_graph'][name] = per_graph.cpu().tolist()
        result.setdefault('output_correct', 0.0)
        result.setdefault('output_total', 0.0)
        result['output_correct'] += float(correct.sum().item())
        result['output_total'] += float(mask.sum().item())
        result.setdefault('per_position_correct', {})[name] = (
            correct.sum(dim=0).cpu().numpy())
        result.setdefault('per_position_total', {})[name] = (
            mask.sum(dim=0).cpu().numpy())

    for name, predictions in output['hint_preds'].items():
        typ = spec[name][2]
        hint = batch['hints'][name]
        correct_total = 0.0
        count_total = 0.0
        se_total = 0.0
        per_step_accuracy = []
        for step, prediction in enumerate(predictions):
            target_step = step + 1
            if target_step >= hint['target'].shape[1]:
                break
            target = hint['target'][:, target_step]
            mask = hint['mask'][:, target_step]
            if typ == specs.Type.SCALAR:
                safe_target = torch.where(
                    mask > 0, target.float(), torch.zeros_like(target.float()))
                diff = (prediction.float() - safe_target) ** 2
                se_total += float((diff * mask.float()).sum().item())
                count_total += float(mask.float().sum().item())
                per_step_accuracy.append(None)
                continue
            correct, mask = _accuracy_counts(prediction, target, mask, typ)
            if correct is None:
                per_step_accuracy.append(None)
                continue
            correct_total += float(correct.sum().item())
            count_total += float(mask.sum().item())
            per_step_accuracy.append(
                float(correct.sum().item() / max(mask.sum().item(), 1.0)))
        result['hints'][name] = {
            'accuracy': (correct_total / count_total
                         if typ != specs.Type.SCALAR and count_total > 0
                         else None),
            'mse': (se_total / count_total
                    if typ == specs.Type.SCALAR and count_total > 0
                    else None),
            'count': count_total,
            'per_step_accuracy': per_step_accuracy,
        }

    gate_values = output.get('gate_values')
    if gate_values:
        result['gate_trajectory'] = [
            float(gate.float().mean().item()) for gate in gate_values]
    return result


def merge_chunks(chunks):
    merged = {
        'n_graphs': sum(chunk['n_graphs'] for chunk in chunks),
        'per_graph': {},
        'hints': {},
        'output_correct': 0.0,
        'output_total': 0.0,
    }
    for chunk in chunks:
        merged['output_correct'] += chunk['output_correct']
        merged['output_total'] += chunk['output_total']
        for name, values in chunk['per_graph'].items():
            merged['per_graph'].setdefault(name, []).extend(values)
        for name, values in chunk.get('per_position_correct', {}).items():
            merged.setdefault('per_position_correct', {}).setdefault(
                name, np.zeros_like(values))
            merged['per_position_correct'][name] += values
        for name, values in chunk.get('per_position_total', {}).items():
            merged.setdefault('per_position_total', {}).setdefault(
                name, np.zeros_like(values))
            merged['per_position_total'][name] += values
        for name, values in chunk['hints'].items():
            entry = merged['hints'].setdefault(name, {
                'correct': 0.0, 'count': 0.0, 'se': 0.0,
                'per_step_correct': None, 'per_step_count': None,
            })
            if values['accuracy'] is None and values['mse'] is None:
                continue
            if values['mse'] is not None:
                entry['se'] += values['mse'] * values['count']
            else:
                entry['correct'] += values['accuracy'] * values['count']
            entry['count'] += values['count']
            per_step = values['per_step_accuracy']
            if values['accuracy'] is not None:
                if entry['per_step_correct'] is None:
                    entry['per_step_correct'] = np.zeros(len(per_step))
                    entry['per_step_count'] = np.zeros(len(per_step))
                for step, value in enumerate(per_step):
                    if value is None:
                        continue
                    entry['per_step_correct'][step] += value
                    entry['per_step_count'][step] += 1
        if 'gate_trajectory' in chunk:
            trajectory = np.asarray(chunk['gate_trajectory'])
            if 'gate_trajectory' not in merged:
                merged['gate_trajectory'] = np.zeros_like(trajectory)
                merged['gate_chunks'] = 0
            merged['gate_trajectory'] += trajectory
            merged['gate_chunks'] += 1

    merged['output_accuracy'] = (
        merged['output_correct'] / max(merged['output_total'], 1.0))
    for name, values in merged.get('per_position_correct', {}).items():
        totals = merged['per_position_total'][name]
        merged.setdefault('per_position_accuracy', {})[name] = (
            values / np.maximum(totals, 1.0)).tolist()
    merged.pop('per_position_correct', None)
    merged.pop('per_position_total', None)
    for name, entry in merged['hints'].items():
        if entry['per_step_correct'] is not None:
            entry['per_step_accuracy'] = (
                entry['per_step_correct']
                / np.maximum(entry['per_step_count'], 1.0)).tolist()
        if entry['se'] > 0:
            entry['mse'] = entry['se'] / max(entry['count'], 1.0)
        else:
            entry['accuracy'] = (
                entry['correct'] / max(entry['count'], 1.0))
        del entry['correct']
        del entry['se']
        del entry['per_step_correct']
        del entry['per_step_count']
    if 'gate_trajectory' in merged:
        merged['gate_trajectory'] = (
            merged['gate_trajectory'] / merged['gate_chunks']).tolist()
        merged['gate_final_mean'] = merged['gate_trajectory'][-1]
        del merged['gate_chunks']
    return merged


def evaluate_batch(model, batch, batch_size, device):
    chunks = []
    total = int(batch['lengths'].numel())
    for start in range(0, total, batch_size):
        chunk = slice_batch(batch, start, min(start + batch_size, total))
        chunk = _to_device(chunk, device)
        chunks.append(eval_chunk(model, chunk))
    return merge_chunks(chunks)


def load_model(args, arch, seed, spec, categorical_dims, device):
    path = checkpoint_path(args, arch, seed)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    model = SpecCLRSNet(
        spec,
        categorical_dims,
        hidden_dim=cfg['hidden_dim'],
        nb_msg_passing_steps=1,
        processor_type=arch,
        gate_init=cfg.get('gate_init', -2.0),
        use_ln=cfg.get('use_ln', True),
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, path


def main():
    args = parse_args()
    if args.smoke:
        args.lengths = [16]
        args.samples = 8
        args.extended_samples = 8
        args.extended_length = 16
        args.batch_size = 8
        args.seeds = [0]
        args.architectures = ['mpnn']
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    started = time.time()

    base_batch = load_fixed_batch(args, 16, args.samples)
    spec = base_batch['spec']
    categorical_dims = base_batch['categorical_dims']

    fixed_batches = {
        length: load_fixed_batch(args, length, args.samples)
        for length in args.lengths
    }
    extended_batch = load_fixed_batch(
        args, args.extended_length, args.extended_samples)

    results = {
        'meta': {
            'algorithm': args.algorithm,
            'architectures': args.architectures,
            'seeds': args.seeds,
            'lengths': args.lengths,
            'samples': args.samples,
            'extended_samples': args.extended_samples,
            'extended_length': args.extended_length,
            'test_seed': args.test_seed,
            'random_pos': False,
            'device': str(device),
            'checkpoint_dir': str(args.checkpoint_dir),
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'length_sweep': {},
        'extended': {},
    }

    for arch in args.architectures:
        print(f'architecture={arch}', flush=True)
        for seed in args.seeds:
            model, path = load_model(
                args, arch, seed, spec, categorical_dims, device)
            print(f'  seed={seed} checkpoint={path.name}', flush=True)
            run_key = f'{arch}/seed{seed}'
            for length in args.lengths:
                metrics = evaluate_batch(
                    model, fixed_batches[length], args.batch_size, device)
                results['length_sweep'].setdefault(
                    str(length), {})[run_key] = metrics
                print(
                    f'    length={length} out={metrics["output_accuracy"]:.4f} '
                    f'hint={_mean_hint(metrics):.4f}', flush=True)
            metrics = evaluate_batch(
                model, extended_batch, args.batch_size, device)
            results['extended'][run_key] = metrics
            print(
                f'    extended={args.extended_length}/{args.extended_samples} '
                f'out={metrics["output_accuracy"]:.4f} '
                f'hint={_mean_hint(metrics):.4f}', flush=True)

    results['meta']['seconds'] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f'wrote {args.output}', flush=True)
    print_summary(results)


def _mean_hint(metrics):
    values = [entry['accuracy'] for entry in metrics['hints'].values()
              if entry.get('accuracy') is not None]
    return float(np.mean(values)) if values else 0.0


def _seed_mean(results, section, length, arch):
    values = [
        metrics['output_accuracy']
        for key, metrics in results[section].get(str(length), {}).items()
        if key.startswith(arch + '/')
    ]
    return float(np.mean(values)) if values else float('nan')


def print_summary(results):
    print('\nlength sweep output accuracy (mean over seeds):')
    header = 'length  ' + '  '.join(
        f'{arch:>10}' for arch in results['meta']['architectures'])
    print(header)
    for length in results['meta']['lengths']:
        values = '  '.join(
            f'{_seed_mean(results, "length_sweep", length, arch):10.4f}'
            for arch in results['meta']['architectures'])
        print(f'{length:>6}  {values}')
    print('\nextended output accuracy:')
    for arch in results['meta']['architectures']:
        values = [
            metrics['output_accuracy']
            for key, metrics in results['extended'].items()
            if key.startswith(arch + '/')
        ]
        print(f'{arch:>10}  {np.mean(values):.4f}')


if __name__ == '__main__':
    main()
