#!/bin/bash
#SBATCH --job-name=coin_open_probe
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/open_probe_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/open_probe_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
SOURCE_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_toy_v1"
EXPECTED_TASKS_SHA256=2e0f65d6801b96616fb53b764470960a96b4c08c73483006ca626477de30ccb8
EXPECTED_TASK_MANIFEST_SHA256=4b73ac0e9634d144feead8a35668f4ddd85c6c6ac8ab0207ef2d17f01d4c4ecb

: "${MODEL:?MODEL is required}"
: "${MODEL_COMMIT:?MODEL_COMMIT is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${STUDY:?STUDY is required}"

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

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then
  echo "Another probe owns $RUN_DIR/.probe.lock; exiting without changes."
  exit 0
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
  --model "$MODEL" \
  --model-id "$MODEL" \
  --model-commit "$MODEL_COMMIT" \
  --model-label "$MODEL_LABEL" \
  --study "$STUDY" \
  --batch-size "${BATCH_SIZE:-1}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-1}" \
  --max-input-tokens 2304 \
  --max-new-tokens 160 \
  --force

"$PY" exp2_v2/biased_news/mechanistic_probe/analyze_probe.py \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --study "$STUDY" \
  --status probe_complete

echo "$MODEL_LABEL mechanistic probe complete: $RUN_DIR"
