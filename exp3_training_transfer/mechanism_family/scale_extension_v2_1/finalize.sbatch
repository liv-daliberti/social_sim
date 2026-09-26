#!/bin/bash
#SBATCH --job-name=c3_scale_v21_probe
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/finalize_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/mechanism_family/scale_extension_v2_1/logs/finalize_%j.err
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail
: "${FULL_LEDGER:?FULL_LEDGER is required}"
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
"$PY" "$REPO/exp3_training_transfer/mechanism_family/scale_extension_v2_1/launch.py" \
  --stage probe --full-ledger "$FULL_LEDGER" --submit
