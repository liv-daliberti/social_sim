#!/bin/bash
#SBATCH --job-name=c3m_final
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/final_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/final_%j.err
#SBATCH --partition=all
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
PY="$REPO/.runtime/oat_conda/bin/python"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
"$PY" "$MECH/analyze_behavior.py"
"$PY" "$MECH/analyze_patching.py"
"$PY" "$MECH/evaluate_gate.py"
echo "Exp3 Qwen3-8B mechanistic gate finalized; no paper files changed"
