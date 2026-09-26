#!/bin/bash
#SBATCH --job-name=cc_q14_symbol_probe_v2
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_probe_v2_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_probe_v2_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_14b_symbol_probe_v2"
EXPECTED_TASKS_SHA256=ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01
EXPECTED_TASK_MANIFEST_SHA256=eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56

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
  --study qwen3_14b_symbol_probe_v2 \
  --batch-size 1 \
  --generation-batch-size 8 \
  --max-input-tokens 2304 \
  --max-new-tokens 160 \
  --force

"$PY" exp2_v2/biased_news/mechanistic_probe/analyze_probe.py \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --study qwen3_14b_symbol_probe_v2 \
  --status probe_complete

echo "Qwen3-14B symbol_probe_v2 complete: $RUN_DIR"
