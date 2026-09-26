#!/bin/bash
#SBATCH --job-name=exp4_qwen32_test
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_test_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/qwen32_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=140G
#SBATCH --cpus-per-task=10
#SBATCH --partition=all
#SBATCH --account=mltheory

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${ADAPTER_SPECS:?ADAPTER_SPECS is required}"
if [ "${MODEL_KEY:-}" != qwen3_32b ] || [ "${MODEL:-}" != Qwen/Qwen3-32B ]; then
  echo "locked Qwen3-32B evaluation received the wrong model" >&2
  exit 2
fi
if [ "${PROMPT_TEMPLATE:-}" != auto_no_think ]; then
  echo "locked Qwen3-32B evaluation requires non-thinking decoding" >&2
  exit 2
fi

OUT="$POLY/reports/exp4_scale_qwen3_32b_stochastic_j${SLURM_JOB_ID:-0}"
"$PY" "$POLY/scripts/evaluate_qwen32_holdout.py" \
  --adapters "$ADAPTER_SPECS" \
  --model-key "$MODEL_KEY" \
  --model "$MODEL" \
  --template "$PROMPT_TEMPLATE" \
  --output-prefix "$OUT"

echo "Qwen3-32B scale evaluation: $OUT"
