#!/bin/bash
#SBATCH --job-name=cc_qwen3_32b_symbol_probe_v2
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_32b_symbol_probe_v2_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_32b_symbol_probe_v2_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00

set -uo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_32b_symbol_probe_v2"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
mkdir -p "$MECH/logs"

if [[ -f "$RUN_DIR/probe_results.json" ]]; then
  echo "Qwen3-32B symbol probe already complete"
  exit 0
fi
exec 9>"$RUN_DIR/.probe.lock"
flock -n 9 || { echo "another allocation owns the probe lock"; exit 0; }
verify() {
  local observed
  observed=$(sha256sum "$1" | cut -d' ' -f1)
  [[ "$observed" == "$2" ]] || { echo "hash mismatch: $1" >&2; exit 1; }
}
verify "$RUN_DIR/tasks.jsonl" ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01
verify "$RUN_DIR/task_manifest.json" eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56
[[ -f "$RUN_DIR/probe_protocol_manifest.json" ]] || { echo "missing frozen protocol" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

"$PY" "$MECH/extract_qwen_hidden_states.py" \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-32B --model-id Qwen/Qwen3-32B \
  --model-commit 9216db5781bf21249d130ec9da846c4624c16137 \
  --model-label Qwen3-32B --study qwen3_32b_symbol_probe_v2 \
  --batch-size 2 --generation-batch-size 4 \
  --max-input-tokens 2304 --max-new-tokens 160 --checkpoint-every 50 \
  --device-map balanced --gpu-memory-gib 44
rc=$?

if [[ -f "$RUN_DIR/extraction_manifest.json" ]]; then
  "$PY" "$MECH/analyze_probe.py" --run-dir "$RUN_DIR" \
    --permutations 10000 --study qwen3_32b_symbol_probe_v2 --status probe_complete
  echo "Qwen3-32B symbol probe v2 complete"
else
  echo "probe stage incomplete (rc=$rc); checkpoint retained for resubmission"
fi
exit 0
