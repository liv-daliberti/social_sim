#!/usr/bin/env bash
#SBATCH --job-name=cc_robust
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/data/coin_city_population_075_040_sd010_v1/logs/%x_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/data/coin_city_population_075_040_sd010_v1/logs/%x_%j.err

set -euo pipefail
REPO=/n/fs/similarity/social_sim
: "${MODEL:?MODEL is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${OUTDIR:?OUTDIR is required}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE is required}"
: "${CHAT_TEMPLATE_MODE:?CHAT_TEMPLATE_MODE is required}"
: "${MODEL_LICENSE_FAMILY:?MODEL_LICENSE_FAMILY is required}"
. "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$REPO/.runtime/oat_conda/bin/python" \
  "$REPO/exp2_v2/biased_news/eval/run_coin_city_robustness_open_model.py" \
  --model "$MODEL" \
  --model-label "$MODEL_LABEL" \
  --outdir "$OUTDIR" \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE" \
  --model-license-family "$MODEL_LICENSE_FAMILY"
