#!/bin/bash
#SBATCH --job-name=exp4_arch_eval
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/arch_test_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/arch_test_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=48G
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
: "${MODEL_CHECKPOINT:?MODEL_CHECKPOINT is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
case "$MODEL_KEY:$MODEL_CHECKPOINT:$PROMPT_TEMPLATE" in
  qwen3_1_7b:Qwen/Qwen3-1.7B:auto_no_think) ;;
  llama3_2_3b:unsloth/Llama-3.2-3B-Instruct:auto) ;;
  *) echo "unregistered architecture evaluation" >&2; exit 2 ;;
esac
OUT="$POLY/reports/exp4_scale_${MODEL_KEY}_stochastic_j${SLURM_JOB_ID:-0}"
"$PY" "$POLY/scripts/evaluate_scale_holdout.py" \
  --adapters "$TRAIN_ADAPTERS" \
  --model-key "$MODEL_KEY" \
  --model "$MODEL" \
  --template "$PROMPT_TEMPLATE" \
  --output-prefix "$OUT"
echo "architecture evaluation: $OUT"
