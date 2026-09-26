#!/bin/bash
#SBATCH --job-name=c3m_seed
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/seed_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/seed_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
EVAL="$REPO/exp3_training_transfer/mechanism_family/evaluate_endpoint.py"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${SEED:?SEED is required}"
: "${ARM:?ARM is required}"
case "$ARM" in matched|prior|base) ;; *) echo "bad ARM=$ARM"; exit 2;; esac
RUN_DIR="$MECH/runs/qwen3_8b_s${SEED}/$ARM"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
mkdir -p "$MECH/logs" "$RUN_DIR"
exec 9>"$RUN_DIR/.seed.lock"
flock -n 9 || { echo "seed $SEED already owned by another worker"; exit 0; }
nvidia-smi
if [[ "$ARM" == "base" ]]; then
  "$PY" "$MECH/extract_states.py" --seed "$SEED" --batch-size 2 \
    --run-dir "$RUN_DIR" --arms --include-base
else
  "$PY" "$MECH/extract_states.py" --seed "$SEED" --batch-size 2 \
    --run-dir "$RUN_DIR" --arms "$ARM"
fi
echo "Exp3 mechanism seed $SEED endpoint $ARM extraction complete"
