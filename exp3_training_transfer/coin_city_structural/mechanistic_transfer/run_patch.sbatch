#!/bin/bash
#SBATCH --job-name=c3m_patch
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/patch_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/patch_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${SEED:?SEED is required}"
RUN_DIR="$MECH/runs/qwen3_8b_s${SEED}"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
mkdir -p "$MECH/logs" "$RUN_DIR"
LOCK_SCOPE=${SLURM_ARRAY_TASK_ID:-smoke}
exec 9>"$RUN_DIR/.patch_${LOCK_SCOPE}.lock"
flock -n 9 || { echo "patch seed $SEED shard $LOCK_SCOPE already owned"; exit 0; }
nvidia-smi
limit_args=()
if [[ -n "${PATCH_LIMIT:-}" ]]; then limit_args=(--limit "$PATCH_LIMIT"); fi
shard_args=()
if [[ -n "${PATCH_SHARDS:-}" ]]; then
  shard_args=(--pair-shard "${SLURM_ARRAY_TASK_ID:?}/${PATCH_SHARDS}")
fi
"$PY" "$MECH/patch_activations.py" --seed "$SEED" --run-dir "$RUN_DIR" \
  "${limit_args[@]}" "${shard_args[@]}"
echo "Exp3 mechanism patch seed $SEED complete"
