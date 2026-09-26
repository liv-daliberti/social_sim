#!/bin/bash
#SBATCH --job-name=exp4_noprice
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/price_ablation_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/price_ablation_%j.err
#SBATCH --partition=all

# Runs every price-ablation condition for one model. Submit with, e.g.
#   sbatch --gres=gpu:a6000:1 --mem=100G --cpus-per-task=8 --time=06:00:00 \
#     --export=ALL,MODEL=...,MODEL_KEY=...,PROMPT_TEMPLATE=...,ADAPTER_SPECS=...,TP=1,RUNS="..." \
#     price_ablation.sbatch
# RUNS is a space-separated list of holdout:decoder:condition triples.

set -euo pipefail
REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL:?}" "${MODEL_KEY:?}" "${PROMPT_TEMPLATE:?}" "${ADAPTER_SPECS:?}" "${RUNS:?}"
TP="${TP:-1}"

for run in $RUNS; do
  IFS=: read -r holdout decoder condition <<<"$run"
  OUT="$POLY/reports/exp4_price_ablation_${MODEL_KEY}_${holdout}_${decoder}_${condition}_j${SLURM_JOB_ID:-0}"
  echo "=== $MODEL_KEY $holdout $decoder $condition -> $OUT"
  "$PY" "$POLY/scripts/evaluate_no_price.py" \
    --model "$MODEL" --model-key "$MODEL_KEY" --template "$PROMPT_TEMPLATE" \
    --adapters "$ADAPTER_SPECS" --holdout "$holdout" --decoder "$decoder" \
    --condition "$condition" --tensor-parallel "$TP" --output-prefix "$OUT"
done
