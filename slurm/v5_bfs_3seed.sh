#!/bin/bash
#SBATCH --job-name=hypare_v5_bfs3
#SBATCH --output=logs/slurm/v5_bfs_%j.out
#SBATCH --error=logs/slurm/v5_bfs_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=ub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home2/home/chakilam_sai/dl/claude/.venv/bin/python}
cd "$PROJECT"
mkdir -p logs/slurm logs/v5 results checkpoints_v5/bfs
export PYTHONPATH="$PROJECT"

echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Started: $(date)"

for arch in mpnn mpnn_gru hypare; do
  for seed in 0 1 2; do
    LOG="$PROJECT/logs/v5/bfs_${arch}_seed${seed}_${SLURM_JOB_ID}.log"
    echo "=== BFS ${arch} seed ${seed} | $(date) ===" >> "$LOG"
    "$PYTHON" -u scripts/train_bfs.py \
      --architectures "$arch" \
      --steps 10000 \
      --eval-every 50 \
      --seeds "$seed" \
      --device cuda \
      --teacher-forcing 0.5 \
      --use-ln \
      --random-pos \
      --output "$PROJECT/results/v5_bfs_${arch}_seed${seed}.json" \
      --checkpoint-dir "$PROJECT/checkpoints_v5/bfs" \
      >> "$LOG" 2>&1
    echo "=== done ${arch} seed ${seed} | $(date) ===" >> "$LOG"
  done
done

echo "Completed: $(date)"
