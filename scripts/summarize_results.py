#!/usr/bin/env python3
"""Print a compact table from a HyPARE experiment JSON file."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f'Results file does not exist: {args.results}')

    payload = json.loads(args.results.read_text())
    summary = payload.get('summary')
    if not summary:
        raise SystemExit(f'No final summary found in {args.results}')

    print('\nFINAL OOD RESULTS')
    print('=' * 60)
    print(f"{'Architecture':<16} {'Mean':>10} {'Std':>10}  Seeds")
    print('-' * 60)
    for architecture, metrics in summary.items():
        scores = ', '.join(f'{score:.4f}' for score in metrics['test_scores'])
        print(f"{architecture:<16} {metrics['test_mean']:>10.4f} "
              f"{metrics['test_std']:>10.4f}  [{scores}]")
    print('=' * 60)
    print(f'Source: {args.results.resolve()}')


if __name__ == '__main__':
    main()
