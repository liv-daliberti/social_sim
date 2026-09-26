#!/bin/bash
#SBATCH --job-name=exp4_scale_test
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_test_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/scale_test_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${ADAPTER_SPECS:?ADAPTER_SPECS is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
OUT="$POLY/reports/exp4_scale_${MODEL_KEY}_stochastic_j${SLURM_JOB_ID:-0}"

"$PY" "$POLY/scripts/evaluate_scale_holdout.py" \
  --adapters "$ADAPTER_SPECS" \
  --model-key "$MODEL_KEY" \
  --model "$MODEL" \
  --template "$PROMPT_TEMPLATE" \
  --output-prefix "$OUT"

echo "scale evaluation: $OUT"
