#!/bin/bash
#SBATCH --job-name=c3_coin_greedy
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/greedy_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/greedy_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
MECHANISM="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${CANARY_JOB_ID:?CANARY_JOB_ID is required}"
: "${ARM:?ARM is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
SEED="${SEED:-42}"

case "$ARM" in causal|population_prior) ;; *) echo "bad canary arm: $ARM"; exit 2;; esac
mapfile -t REPORTS < <(find "$FAMILY/reports" -maxdepth 1 -type d \
  -name "canary_${ARM}_${MODEL_KEY}_s${SEED}_*_j${CANARY_JOB_ID}" -print)
[ "${#REPORTS[@]}" -eq 1 ] || {
  echo "expected one canary report for job $CANARY_JOB_ID, got ${#REPORTS[@]}"; exit 3;
}
REPORT="${REPORTS[0]}"
mapfile -t ADAPTERS < <(find "$REPORT" -type f \
  -path '*/saved_models/step_*/adapter_model.safetensors' -print)
[ "${#ADAPTERS[@]}" -eq 1 ] && [ -s "${ADAPTERS[0]}" ] || {
  echo "expected one nonempty adapter for job $CANARY_JOB_ID"; exit 4;
}
ADAPTER="$(dirname "${ADAPTERS[0]}")"
DATA="$FAMILY/data/$ARM/heldout"

"$PY" "$MECHANISM/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" --data "$DATA" \
  --template "$PROMPT_TEMPLATE" --structured-output forecast_array \
  --temperature 0 --n 1 --seed "$(( SEED + 20260818 ))" \
  --max-tokens 192 --max-model-len 3072 --output "$REPORT/greedy.json"
"$PY" "$FAMILY/report.py" score "$REPORT/greedy.json" \
  --model "$MODEL_KEY" --arm "$ARM" --seed "$SEED"
echo "constrained greedy endpoint complete: $REPORT/greedy.json"
