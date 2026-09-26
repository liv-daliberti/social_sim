#!/bin/bash
#SBATCH --job-name=coin_q14_symprobe
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_probe_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_probe_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_14b_symbol_probe_v1"
EXPECTED_TASKS_SHA256=6d4dd81bb031d7dc00320e97d2c1896a132c3deab54a6642b6b0105a954014a9
EXPECTED_TASK_MANIFEST_SHA256=35cf4745a92fa10def497662156edcc9f16f98c8f36bfac19e72a986c9020a2c

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

exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then
  echo "Another probe owns $RUN_DIR/.probe.lock; exiting without changes."
  exit 0
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
  --model Qwen/Qwen3-14B \
  --model-label Qwen3-14B \
  --study qwen3_14b_symbol_probe_v1 \
  --batch-size 1 \
  --generation-batch-size 2 \
  --max-input-tokens 2304 \
  --max-new-tokens 160 \
  --force

"$PY" exp2_v2/biased_news/mechanistic_probe/analyze_probe.py \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --study qwen3_14b_symbol_probe_v1 \
  --status probe_complete

echo "Qwen3-14B arbitrary-symbol mechanistic probe complete: $RUN_DIR"
