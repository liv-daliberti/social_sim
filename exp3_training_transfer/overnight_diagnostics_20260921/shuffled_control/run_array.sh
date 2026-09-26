#!/bin/bash
set -euo pipefail
REPO=/n/fs/similarity/social_sim
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control"
source "$REPO/.runtime/oat_env.sh"
exec "$REPO/.runtime/oat_conda/bin/python" "$ROOT/launch.py" --run-cell "${1:?roster kind required}" --index "${SLURM_ARRAY_TASK_ID:?array task required}"
