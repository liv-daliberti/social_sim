#!/bin/bash
#SBATCH --job-name=structure_diag
#SBATCH --partition=all,cs
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=node206
set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
export OMP_NUM_THREADS=4
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/structure"
"$REPO/.runtime/oat_conda/bin/python" "$ROOT/evaluate.py" --model "${MODEL:-Qwen/Qwen3-8B}" --tensor-parallel-size "${TP:-1}"
