#!/bin/bash
#SBATCH --job-name=c3_coin_base
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/base_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/base_%j.err
#SBATCH --partition=cs
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
MECHANISM="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TEMPLATE="${PROMPT_TEMPLATE:-biased_news}"
DATA="$FAMILY/data/causal/heldout"
OUT="$FAMILY/reports/base_${MODEL_KEY}_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$FAMILY/logs"

"$PY" "$MECHANISM/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature 0 --n 1 --seed 20260818 \
  --structured-output forecast_array --max-tokens 192 --max-model-len 3072 \
  --output "$OUT/greedy.json"
"$PY" "$MECHANISM/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature 0.7 --n 5 --seed 20260819 \
  --structured-output forecast_array --max-tokens 192 --max-model-len 3072 \
  --output "$OUT/stochastic_n5.json"
"$PY" "$FAMILY/report.py" score "$OUT/greedy.json" --model "$MODEL_KEY" --arm base --seed 0
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n5.json" --model "$MODEL_KEY" --arm base --seed 0
echo "complete: $OUT"
