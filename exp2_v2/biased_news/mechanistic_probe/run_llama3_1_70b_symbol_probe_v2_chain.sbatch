#!/bin/bash
#SBATCH --job-name=cc_l70_chain
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_chain_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_chain_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00

set -uo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/llama3_1_70b_symbol_probe_v2"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} TOKENIZERS_PARALLELISM=false

if [[ -f "$RUN_DIR/probe_results.json" ]]; then
  echo "run already complete; nothing to do"; exit 0
fi

exec 9>"$RUN_DIR/.probe.lock"
if ! flock -n 9; then echo "another allocation owns the lock"; exit 0; fi

verify() { o=$(sha256sum "$1"|cut -d' ' -f1); [[ "$o" == "$2" ]] || { echo "hash mismatch $1" >&2; exit 1; }; }
verify "$RUN_DIR/tasks.jsonl" ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01
verify "$RUN_DIR/task_manifest.json" eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

# Resumable: a killed allocation leaves a checkpoint that the next one continues.
"$PY" $REPO/exp2_v2/biased_news/mechanistic_probe/extract_qwen_hidden_states.py \
  --run-dir "$RUN_DIR" \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --model-commit 1605565b47bb9346c5515c34102e054115b4f98b \
  --model-label Llama-3.1-70B-Instruct \
  --study llama3_1_70b_symbol_probe_v2 \
  --batch-size 4 --generation-batch-size 8 \
  --max-input-tokens 2304 --max-new-tokens 160 \
  --device-map balanced --gpu-memory-gib 44 \
  --checkpoint-every 50
rc=$?

if [[ -f "$RUN_DIR/extraction_manifest.json" ]]; then
  "$PY" $REPO/exp2_v2/biased_news/mechanistic_probe/analyze_probe.py --run-dir "$RUN_DIR" \
    --permutations 10000 --study llama3_1_70b_symbol_probe_v2 --status probe_complete
  echo "Llama-3.1-70B symbol probe v2 complete"
else
  echo "stage incomplete (extract rc=$rc); checkpoint retained for the next allocation"
fi
exit 0
