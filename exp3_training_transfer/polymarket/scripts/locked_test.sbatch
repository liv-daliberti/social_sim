#!/bin/bash
#SBATCH --job-name=exp3b_test
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/locked_test_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/locked_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

if [[ -z "${TRAIN_ADAPTERS:-}" ]]; then
    echo "TRAIN_ADAPTERS must be seed:job_id pairs" >&2
    exit 2
fi

OUT="$POLY/reports/exp3b_locked_test_j${SLURM_JOB_ID:-0}"
"$PY" "$POLY/scripts/evaluate_locked_test.py" \
    --adapters "$TRAIN_ADAPTERS" \
    --output-prefix "$OUT"

echo "locked-test summary: ${OUT}.summary.json"
