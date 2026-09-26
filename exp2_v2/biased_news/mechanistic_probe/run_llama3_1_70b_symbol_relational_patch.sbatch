#!/bin/bash
#SBATCH --job-name=cc_l70_rel_patch
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_patch_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/l70_rel_patch_%j.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
RUN_DIR="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/llama3_1_70b_symbol_relational_v1"
: "${DEPTH:?DEPTH must be 0 or 4}"
case "$DEPTH" in 0|4) ;; *) echo "invalid DEPTH=$DEPTH" >&2; exit 2;; esac
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$MECH/logs"
exec 9>"$RUN_DIR/.patch_depth_${DEPTH}.lock"
flock -n 9 || { echo "Llama relational patch depth $DEPTH already owned"; exit 0; }
if [[ -f "$RUN_DIR/activation_patch_manifest.json" ]]; then
  echo "Llama relational patch run already complete"; exit 0
fi
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PY" "$MECH/patch_symbol_label_activations.py" \
  --run-dir "$RUN_DIR" \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --device-map balanced --gpu-memory-gib 44 --depths "$DEPTH"
echo "Llama-3.1-70B relational patch depth $DEPTH complete"
