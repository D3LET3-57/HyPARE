#!/bin/bash
#SBATCH --job-name=hypare_v5_dfs_base
#SBATCH --output=logs/slurm/v5_dfs_base_%j.out
#SBATCH --error=logs/slurm/v5_dfs_base_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=ub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home2/home/chakilam_sai/dl/claude/.venv/bin/python}
cd "$PROJECT"
mkdir -p logs/slurm logs/v5 results checkpoints_v5
export PYTHONPATH="$PROJECT"

echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Started: $(date)"

configs=("official:--no-use-ln" "official_ln:--use-ln")

for cfg in "${configs[@]}"; do
  label=${cfg%%:*}
  lnflag=${cfg#*:}
  for seed in 0 1 2; do
    LOG="$PROJECT/logs/v5/dfs_${label}_seed${seed}_${SLURM_JOB_ID}.log"
    echo "=== DFS MPNN ${label} seed ${seed} | $(date) ===" >> "$LOG"
    "$PYTHON" -u scripts/train_clrs.py \
      --algorithm dfs \
      --architectures mpnn \
      --steps 10000 \
      --eval-every 500 \
      --test-every 0 \
      --seeds "$seed" \
      --device cuda \
      --teacher-forcing 0.5 \
      --no-random-pos \
      "$lnflag" \
      --train-length 16 \
      --val-length 16 \
      --test-length 64 \
      --train-samples 1000 \
      --val-samples 32 \
      --test-samples 32 \
      --output "$PROJECT/results/v5_dfs_${label}_seed${seed}.json" \
      --checkpoint-dir "$PROJECT/checkpoints_v5/${label}" \
      >> "$LOG" 2>&1
    echo "=== done ${label} seed ${seed} | $(date) ===" >> "$LOG"
  done
done

echo "Completed: $(date)"
