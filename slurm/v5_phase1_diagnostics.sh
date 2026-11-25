#!/bin/bash
#SBATCH --job-name=hypare_v5_diag
#SBATCH --output=logs/slurm/v5_diag_%j.out
#SBATCH --error=logs/slurm/v5_diag_%j.err
#SBATCH --time=04:00:00
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

LOG="$PROJECT/logs/v5/dfs_diagnostics_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo "HyPARE v5 Phase 1 diagnostics"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Log: $LOG"
echo "=================================================================="

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is unavailable on this SLURM allocation.')
print('PyTorch:', torch.__version__)
print('CUDA device:', torch.cuda.get_device_name(0))
PY

"$PYTHON" -u scripts/diagnose_dfs.py \
  --algorithm dfs \
  --architectures mpnn mpnn_gru hypare \
  --seeds 0 1 2 \
  --lengths 16 24 32 48 64 \
  --samples 32 \
  --extended-samples 256 \
  --extended-length 64 \
  --batch-size 32 \
  --test-seed 3 \
  --device cuda \
  --checkpoint-dir "$PROJECT/checkpoints_v4" \
  --output "$PROJECT/results/v5_dfs_diagnostics.json"

echo "Completed: $(date)"
