#!/bin/bash
#SBATCH --job-name=exp3b_ext_test
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/ext_test_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/ext_test_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${TRAIN_ADAPTERS:?TRAIN_ADAPTERS is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
OUT="$POLY/reports/exp3b_${MODEL_KEY}_locked_test_j${SLURM_JOB_ID:-0}"
"$PY" "$POLY/scripts/evaluate_locked_test_model_extension.py" \
  --adapters "$TRAIN_ADAPTERS" --model-key "$MODEL_KEY" --model "$MODEL" \
  --template "$PROMPT_TEMPLATE" --output-prefix "$OUT"
