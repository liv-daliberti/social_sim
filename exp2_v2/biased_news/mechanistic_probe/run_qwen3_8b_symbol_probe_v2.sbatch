#!/bin/bash
#SBATCH --job-name=cc_qwen3_8b_symbol_probe_v2
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_8b_symbol_probe_v2_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen3_8b_symbol_probe_v2_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -uo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_8b_symbol_probe_v2"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} TOKENIZERS_PARALLELISM=false

if [[ -f "$RUN_DIR/probe_results.json" ]]; then echo "already complete"; exit 0; fi
exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then echo "another allocation owns the lock"; exit 0; fi
verify() { o=$(sha256sum "$1"|cut -d' ' -f1); [[ "$o" == "$2" ]] || { echo "hash mismatch $1" >&2; exit 1; }; }
verify "$RUN_DIR/tasks.jsonl" ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01
verify "$RUN_DIR/task_manifest.json" eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56
nvidia-smi --query-gpu=index,name --format=csv,noheader

"$PY" $REPO/exp2_v2/biased_news/mechanistic_probe/extract_qwen_hidden_states.py \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-8B --model-id Qwen/Qwen3-8B \
  --model-commit b968826d9c46dd6066d109eabc6255188de91218 --model-label Qwen3-8B \
  --study qwen3_8b_symbol_probe_v2 \
  --batch-size 4 --generation-batch-size 8 \
  --max-input-tokens 2304 --max-new-tokens 160 \
  --checkpoint-every 50 
rc=$?

if [[ -f "$RUN_DIR/extraction_manifest.json" ]]; then
  "$PY" $REPO/exp2_v2/biased_news/mechanistic_probe/analyze_probe.py --run-dir "$RUN_DIR" \
    --permutations 10000 --study qwen3_8b_symbol_probe_v2 --status probe_complete
  echo "Qwen3-8B symbol probe v2 complete"
else
  echo "stage incomplete (rc=$rc); checkpoint retained for the next allocation"
fi
exit 0
