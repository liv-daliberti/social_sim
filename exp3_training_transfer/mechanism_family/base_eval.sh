#!/bin/bash
#SBATCH --job-name=c3_base
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/logs/base_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/logs/base_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

DISCLOSURE="${DISCLOSURE:-disclosed}"
MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TEMPLATE="${PROMPT_TEMPLATE:-biased_news}"
MAX_TOKENS="${MAX_TOKENS:-192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
STRUCTURED_OUTPUT="${STRUCTURED_OUTPUT:-forecast_array}"
DATA="$FAMILY/data/${DISCLOSURE}_causal_family/heldout"
OUT="$FAMILY/reports/base_${DISCLOSURE}_${MODEL_KEY}_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"

"$PY" "$FAMILY/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature 0 --n 1 --seed 20260814 \
  --structured-output "$STRUCTURED_OUTPUT" \
  --max-tokens "$MAX_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/greedy.json"
"$PY" "$FAMILY/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature "${STOCHASTIC_TEMP:-0.7}" \
  --n "${STOCHASTIC_N:-5}" --seed 20260815 \
  --structured-output "$STRUCTURED_OUTPUT" \
  --max-tokens "$MAX_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/stochastic_n${STOCHASTIC_N:-5}.json"

"$PY" "$FAMILY/report.py" score "$OUT/greedy.json" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm base --seed 0
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
  --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm base --seed 0

echo "complete: $OUT"
