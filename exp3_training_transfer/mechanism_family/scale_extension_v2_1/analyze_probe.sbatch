#!/bin/bash
#SBATCH --job-name=c3_scale_probe_analysis
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/analysis_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/analysis_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

REPO=/n/fs/similarity/social_sim
PROBE="$REPO/exp3_training_transfer/mechanism_family/mechanistic_probe"
PY="$REPO/.runtime/oat_conda/bin/python"
: "${RUN_DIR:?RUN_DIR is required}"

cd "$REPO"
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

exec 9>"$RUN_DIR/.scale_v2_1_analysis.lock"
flock -n 9 || { echo "another analysis owns $RUN_DIR"; exit 0; }
"$PY" "$PROBE/analyze_probe_v2.py" \
  --task-dir "$PROBE/data" \
  --auxiliary-targets "$PROBE/data/auxiliary_targets_v2.jsonl" \
  --run-dir "$RUN_DIR" \
  --permutations 10000 \
  --status probe_v2_1_complete

echo "large-model analysis complete: $RUN_DIR"
