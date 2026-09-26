#!/bin/bash
#SBATCH --job-name=coin_q_large_probe
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_large_probe_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_large_probe_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
SOURCE_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_toy_v1"

: "${MODEL:?MODEL is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${STUDY:?STUDY is required}"

GPU_MEMORY_GIB=${GPU_MEMORY_GIB:-44}
STAGE_COPY_JOBS=${STAGE_COPY_JOBS:-4}
EXPECTED_TASKS_SHA256=2e0f65d6801b96616fb53b764470960a96b4c08c73483006ca626477de30ccb8
EXPECTED_TASK_MANIFEST_SHA256=4b73ac0e9634d144feead8a35668f4ddd85c6c6ac8ab0207ef2d17f01d4c4ecb

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false

nvidia-smi

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then
  echo "Another probe owns $RUN_DIR/.probe.lock; exiting without changes."
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
  required_bytes=$(du -sbL "$MODEL_SNAPSHOT" | cut -f1)
  available_bytes=$(df -PB1 /tmp | awk 'NR==2 {print $4}')
  reserve_bytes=$((10 * 1024 * 1024 * 1024))
  if (( available_bytes >= required_bytes + reserve_bytes )); then
    STAGE_DIR=$(mktemp -d /tmp/qwen_probe_stage.XXXXXX)
    echo "Staging $MODEL_SNAPSHOT to $STAGE_DIR"
    find -L "$MODEL_SNAPSHOT" -mindepth 1 -maxdepth 1 -type f \
      ! -name '*.safetensors' -exec cp -aL -t "$STAGE_DIR" -- {} +
    find -L "$MODEL_SNAPSHOT" -mindepth 1 -maxdepth 1 -type f \
      -name '*.safetensors' -print0 \
      | xargs -0 -r -P "$STAGE_COPY_JOBS" -n 1 cp -aL -t "$STAGE_DIR" --
    observed_shards=$(find "$STAGE_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
    if [[ "$observed_shards" -ne "${EXPECTED_WEIGHT_SHARDS:?}" ]]; then
      echo "Expected $EXPECTED_WEIGHT_SHARDS shards; staged $observed_shards" >&2
      exit 1
    fi
    MODEL_SOURCE="$STAGE_DIR"
    du -sh "$STAGE_DIR"
  else
    echo "Insufficient /tmp capacity for staging; loading from $MODEL instead."
  fi
fi


if [[ ! -f "$RUN_DIR/tasks.jsonl" ]]; then
  cp -a "$SOURCE_DIR/tasks.jsonl" "$RUN_DIR/tasks.jsonl"
fi
if [[ ! -f "$RUN_DIR/task_manifest.json" ]]; then
  cp -a "$SOURCE_DIR/task_manifest.json" "$RUN_DIR/task_manifest.json"
fi

observed_tasks_sha256=$(sha256sum "$RUN_DIR/tasks.jsonl" | cut -d' ' -f1)
observed_manifest_sha256=$(sha256sum "$RUN_DIR/task_manifest.json" | cut -d' ' -f1)
if [[ "$observed_tasks_sha256" != "$EXPECTED_TASKS_SHA256" ]]; then
  echo "Unexpected tasks.jsonl hash: $observed_tasks_sha256" >&2
  exit 1
fi
if [[ "$observed_manifest_sha256" != "$EXPECTED_TASK_MANIFEST_SHA256" ]]; then
  echo "Unexpected task_manifest.json hash: $observed_manifest_sha256" >&2
  exit 1
fi

"$PY" exp2_v2/biased_news/mechanistic_probe/extract_qwen_hidden_states.py \
  --run-dir "$RUN_DIR" \
  --model "$MODEL_SOURCE" \
  --model-id "$MODEL" \
  --model-commit "$MODEL_COMMIT" \
  --model-label "$MODEL_LABEL" \
  --study "$STUDY" \
  --batch-size 1 \
  --generation-batch-size 1 \
  --max-input-tokens 2304 \
  --max-new-tokens 160 \
  --device-map balanced \
  --gpu-memory-gib "$GPU_MEMORY_GIB" \
  ${SKIP_GENERATION:+--skip-generation} \
  --force

echo "$MODEL_LABEL hidden-state extraction complete: $RUN_DIR"
