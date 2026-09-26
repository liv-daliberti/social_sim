#!/bin/bash
#SBATCH --job-name=c3m_smoke
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/smoke_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/smoke_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
PY="$REPO/.runtime/oat_conda/bin/python"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_TF=1
export USE_TF=0 USE_FLAX=0 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$MECH/logs" "$MECH/runs/smoke_seed42"
nvidia-smi
"$PY" "$MECH/extract_states.py" --seed 42 --limit 2 --include-base \
  --batch-size 1 --run-dir "$MECH/runs/smoke_seed42"
echo "Exp3 mechanism extraction smoke passed"
