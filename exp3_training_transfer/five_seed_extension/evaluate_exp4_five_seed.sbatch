#!/bin/bash
#SBATCH --job-name=exp4_five_seed_eval
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/exp4_eval_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/exp4_eval_%j.err

set -euo pipefail
REPO=/n/fs/similarity/social_sim
EXT="$REPO/exp3_training_transfer/five_seed_extension"
POLY="$REPO/exp3_training_transfer/polymarket"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${ADAPTER_SPECS:?ADAPTER_SPECS is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
source "$REPO/.runtime/oat_env.sh"
mkdir -p "$EXT/logs" "$POLY/reports"
OUT="$POLY/reports/exp4_five_seed_${MODEL_KEY}_stochastic_j${SLURM_JOB_ID:-0}"
"$PY" "$EXT/evaluate_exp4_five_seed.py" \
  --adapters "$ADAPTER_SPECS" --model-key "$MODEL_KEY" --model "$MODEL" \
  --template "$PROMPT_TEMPLATE" --output-prefix "$OUT"
