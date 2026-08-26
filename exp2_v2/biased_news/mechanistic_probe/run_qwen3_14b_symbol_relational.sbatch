#!/bin/bash
#SBATCH --job-name=coin_q14_relpatch
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_relational_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_14b_symbol_relational_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
PROBE="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_14b_symbol_relational_v1"

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

exec 9>"$RUN_DIR/.relational_probe.lock"
if ! flock -n 9; then
  echo "Another run owns $RUN_DIR/.relational_probe.lock; exiting without changes."
  exit 0
fi

verify_hash() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$observed" != "$expected" ]]; then
    echo "Hash mismatch for $path: observed=$observed expected=$expected" >&2
    exit 1
  fi
}

verify_hash "$RUN_DIR/tasks.jsonl" 6d4dd81bb031d7dc00320e97d2c1896a132c3deab54a6642b6b0105a954014a9
verify_hash "$RUN_DIR/task_manifest.json" 35cf4745a92fa10def497662156edcc9f16f98c8f36bfac19e72a986c9020a2c
verify_hash "$RUN_DIR/relational_protocol_manifest.json" 8a002702c134d5547aa6a15ae377abc02968de0d952b20bd66e43abdf15ee0c2
verify_hash "$PROBE/symbol_relational_common.py" d334306f7f9246d145589b38fbd879422855739bffd1b5b03b44a025b7dee00c
verify_hash "$PROBE/extract_symbol_label_states.py" d916a32f3925b36d2e12baf71736931133fdc022109f7884bb72f2560b86914f
verify_hash "$PROBE/analyze_symbol_relational_probe.py" 40e2e1ef637e851264b774c11d82c3867ba7b5ef6ad1a5d35449d1ff321f5494
verify_hash "$PROBE/patch_symbol_label_activations.py" b8d21013664ed539ea8fce18c012bd434676d9f90a94ed1876de5b3c8410722f
verify_hash "$PROBE/freeze_symbol_relational_protocol.py" 5a2fd29484e91ea60f1dd24a13bd8f4495d32cccd2b741129f30691d21d2e8c3

"$PY" "$PROBE/extract_symbol_label_states.py" \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-14B \
  --max-input-tokens 2304

"$PY" "$PROBE/analyze_symbol_relational_probe.py" \
  --mode selection-only \
  --run-dir "$RUN_DIR"

"$PY" "$PROBE/patch_symbol_label_activations.py" \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-14B \
  --max-new-tokens 160

"$PY" "$PROBE/analyze_symbol_relational_probe.py" \
  --mode final \
  --run-dir "$RUN_DIR" \
  --permutations 10000

echo "Qwen3-14B symbolic relational/causal probe complete: $RUN_DIR"
