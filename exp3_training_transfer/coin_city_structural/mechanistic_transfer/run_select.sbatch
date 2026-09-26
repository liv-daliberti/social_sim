#!/bin/bash
#SBATCH --job-name=c3m_select
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/select_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/mechanistic_transfer/logs/select_%j.err
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
MECH="$REPO/exp3_training_transfer/coin_city_structural/mechanistic_transfer"
PY="$REPO/.runtime/oat_conda/bin/python"
cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
"$PY" "$MECH/select_probe.py"
"$PY" "$MECH/evaluate_probe.py"
echo "development selection frozen and sealed common-readout test complete"
