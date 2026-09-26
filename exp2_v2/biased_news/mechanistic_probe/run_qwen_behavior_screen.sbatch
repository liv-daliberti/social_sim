#!/bin/bash
#SBATCH --job-name=coin_q_behavior
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_behavior_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_behavior_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"

: "${MODEL:?MODEL is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${OUT_DIR:?OUT_DIR is required}"

TASKS_DIR=${TASKS_DIR:-$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_toy_v1}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
DEVICE_MAP=${DEVICE_MAP:-auto}
GPU_MEMORY_GIB=${GPU_MEMORY_GIB:-44}

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

nvidia-smi

"$PY" exp2_v2/biased_news/mechanistic_probe/screen_qwen_behavior.py \
  --tasks-dir "$TASKS_DIR" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --model-label "$MODEL_LABEL" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device-map "$DEVICE_MAP" \
  --gpu-memory-gib "$GPU_MEMORY_GIB" \
  --force

echo "Qwen behavior screen complete: $OUT_DIR"
