#!/bin/bash
#SBATCH --job-name=coin_q14_relv2_patch4
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/coin_q14_relv2_patch4_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/coin_q14_relv2_patch4_%j.err
#SBATCH --partition=all
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
PROBE="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_14b_symbol_relational_v2"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} TOKENIZERS_PARALLELISM=false
nvidia-smi
exec 9>"$RUN_DIR/.relational_probe.lock"
if ! flock -n 9; then echo "another run owns the lock; exiting"; exit 0; fi
verify_hash() {
  local observed; observed=$(sha256sum "$1" | cut -d' ' -f1)
  if [[ "$observed" != "$2" ]]; then echo "Hash mismatch for $1: $observed != $2" >&2; exit 1; fi
}
verify_hash "$RUN_DIR/tasks.jsonl" ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01
verify_hash "$RUN_DIR/task_manifest.json" eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56
verify_hash "$RUN_DIR/relational_protocol_manifest.json" 2f33d86907833fe3446077f956165c2cfe9973df6e312e9743f95a9776364b18
verify_hash "$PROBE/symbol_relational_common.py" d334306f7f9246d145589b38fbd879422855739bffd1b5b03b44a025b7dee00c
verify_hash "$PROBE/extract_symbol_label_states.py" d916a32f3925b36d2e12baf71736931133fdc022109f7884bb72f2560b86914f
verify_hash "$PROBE/analyze_symbol_relational_probe.py" 40e2e1ef637e851264b774c11d82c3867ba7b5ef6ad1a5d35449d1ff321f5494
verify_hash "$PROBE/patch_symbol_label_activations.py" b8d21013664ed539ea8fce18c012bd434676d9f90a94ed1876de5b3c8410722f
verify_hash "$PROBE/freeze_symbol_relational_protocol.py" 5a2fd29484e91ea60f1dd24a13bd8f4495d32cccd2b741129f30691d21d2e8c3

"$PY" "$PROBE/patch_symbol_label_activations.py" --run-dir "$RUN_DIR" --model Qwen/Qwen3-14B --max-new-tokens 160 --depths 4
"$PY" "$PROBE/analyze_symbol_relational_probe.py" --mode final --run-dir "$RUN_DIR" --permutations 10000
echo "v2 relational/causal probe complete"
