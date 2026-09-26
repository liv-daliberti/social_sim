#!/bin/bash
#SBATCH --job-name=cc_qwen_rel_patch
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_patch_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/mechanistic_probe/logs/qwen_rel_patch_%j.err
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00

set -euo pipefail
: "${RUN_NAME:?RUN_NAME is required}"
: "${MODEL_ID:?MODEL_ID is required}"
: "${DEPTH:?DEPTH must be 0 or 4}"
case "$DEPTH" in 0|4) ;; *) echo "invalid DEPTH=$DEPTH" >&2; exit 2;; esac

REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
MECH="$REPO/exp2_v2/biased_news/mechanistic_probe"
DATA="$REPO/exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe"
RUN_DIR="$DATA/$RUN_NAME"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$MECH/logs"
exec 9>"$RUN_DIR/.patch_depth_${DEPTH}.lock"
flock -n 9 || { echo "$RUN_NAME patch depth $DEPTH already owned"; exit 0; }
if [[ -f "$RUN_DIR/activation_patch_manifest.json" ]]; then
  echo "$RUN_NAME patch run already complete"
  exit 0
fi
for name in symbol_relational_common.py extract_symbol_label_states.py \
  analyze_symbol_relational_probe.py patch_symbol_label_activations.py \
  freeze_symbol_relational_protocol.py; do
  expected=$(jq -r --arg name "$name" '.code_sha256[$name]' \
    "$RUN_DIR/relational_protocol_manifest.json")
  observed=$(sha256sum "$MECH/$name" | cut -d' ' -f1)
  [[ "$observed" == "$expected" ]] || {
    echo "frozen code hash mismatch: $name" >&2
    exit 1
  }
done
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PY" "$MECH/patch_symbol_label_activations.py" \
  --run-dir "$RUN_DIR" --model "$MODEL_ID" \
  --device-map balanced --gpu-memory-gib 44 --depths "$DEPTH"
echo "$RUN_NAME patch depth $DEPTH complete"
