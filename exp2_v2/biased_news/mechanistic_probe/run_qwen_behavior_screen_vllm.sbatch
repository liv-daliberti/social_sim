#!/bin/bash
#SBATCH --job-name=coin_q72_behavior
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_behavior_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_behavior_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"

: "${MODEL:?MODEL is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${OUT_DIR:?OUT_DIR is required}"

TASKS_DIR=${TASKS_DIR:-$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_toy_v1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-4}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2304}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_FLAX=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi

mkdir -p "$OUT_DIR"
LOCK_FILE="$OUT_DIR/.behavior_screen.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another behavior screen owns $LOCK_FILE; exiting without changes."
  exit 0
fi

if [[ -f "$OUT_DIR/behavior_test.jsonl" || -f "$OUT_DIR/behavior_manifest.json" ]]; then
  "$PY" exp2_v2/biased_news/mechanistic_probe/screen_qwen_behavior_vllm.py \
    --tasks-dir "$TASKS_DIR" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --model-label "$MODEL_LABEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  echo "Existing Qwen behavior screen validated; skipping model staging."
  exit 0
fi

MODEL_SOURCE="$MODEL"
STAGE_DIR=""
cleanup_stage() {
  if [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
}
trap cleanup_stage EXIT

if [[ -n "${MODEL_SNAPSHOT:-}" ]]; then
  if [[ ! -d "$MODEL_SNAPSHOT" ]]; then
    echo "MODEL_SNAPSHOT is not a directory: $MODEL_SNAPSHOT" >&2
    exit 1
  fi
  STAGE_DIR=$(mktemp -d /tmp/qwen_behavior_stage.XXXXXX)
  echo "Staging $MODEL_SNAPSHOT to $STAGE_DIR"
  find -L "$MODEL_SNAPSHOT" -mindepth 1 -maxdepth 1 -type f \
    ! -name '*.safetensors' \
    -exec cp -aL -t "$STAGE_DIR" -- {} +
  find -L "$MODEL_SNAPSHOT" -mindepth 1 -maxdepth 1 -type f \
    -name '*.safetensors' -print0 \
    | xargs -0 -r -P "${STAGE_COPY_JOBS:-4}" -n 1 \
      cp -aL -t "$STAGE_DIR" --
  if [[ $(find "$STAGE_DIR" -maxdepth 1 -name '*.safetensors' | wc -l) -ne 37 ]]; then
    echo "Staged checkpoint does not contain all 37 weight shards" >&2
    exit 1
  fi
  MODEL_SOURCE="$STAGE_DIR"
  du -sh "$STAGE_DIR"
fi

export MODEL_ID="$MODEL"

"$PY" exp2_v2/biased_news/mechanistic_probe/screen_qwen_behavior_vllm.py \
  --tasks-dir "$TASKS_DIR" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL_SOURCE" \
  --model-label "$MODEL_LABEL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"

echo "Qwen vLLM behavior screen complete: $OUT_DIR"
