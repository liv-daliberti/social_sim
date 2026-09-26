#!/bin/bash
#SBATCH --job-name=mech_evid_ext
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/logs/ext_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/logs/ext_%j.err
#SBATCH --account=allcs
#SBATCH --partition=cs
#SBATCH --qos=medium
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
TASK="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use"
"$REPO/.runtime/oat_conda/bin/python" "$TASK/run_inference.py" \
  --disclosure "${DISCLOSURE:?set DISCLOSURE}" --data "$TASK/data_ext" --output "$TASK/results_ext"
