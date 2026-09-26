#!/bin/bash
#SBATCH --job-name=c3_probe_analyze_s45_46
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/probe_analysis_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/five_seed_extension/logs/probe_analysis_%j.err
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
source "$REPO/.runtime/oat_env.sh"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
exec 9>"$RUN_DIR/.five_seed_analysis.lock"
flock -n 9 || { echo "another analysis owns $RUN_DIR"; exit 0; }
"$PY" "$PROBE/analyze_probe_v2.py" \
  --task-dir "$PROBE/data" \
  --auxiliary-targets "$PROBE/data/auxiliary_targets_v2.jsonl" \
  --run-dir "$RUN_DIR" --permutations 10000 \
  --status probe_five_seed_extension_complete
