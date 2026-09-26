#!/bin/bash
#SBATCH --job-name=c3_probe_v2_analysis
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/probe_v2_analysis_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/mechanistic_probe/logs/probe_v2_analysis_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"
TASK_DIR="$PROBE/data"

: "${RUN_DIR:?RUN_DIR is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

mkdir -p "$PROBE/logs"
exec 9>"$RUN_DIR/.probe_v2_analysis.lock"
if ! flock -n 9; then
  echo "Another analysis owns $RUN_DIR/.probe_v2_analysis.lock; exiting."
  exit 0
fi

"$PY" "$PROBE/analyze_probe_v2.py" \
  --task-dir "$TASK_DIR" \
  --auxiliary-targets "$TASK_DIR/auxiliary_targets_v2.jsonl" \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --status probe_v2_complete

echo "Probe v2 analysis complete: $RUN_DIR"
