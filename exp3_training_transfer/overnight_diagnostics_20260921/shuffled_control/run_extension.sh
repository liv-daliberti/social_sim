#!/bin/bash
set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
exec "$REPO/.runtime/oat_conda/bin/python" \
  "$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control/run_extension.py" \
  --index "${SLURM_ARRAY_TASK_ID:?}"
