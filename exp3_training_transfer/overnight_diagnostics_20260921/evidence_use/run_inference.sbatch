#!/bin/bash
#SBATCH --job-name=mech_evidence
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/logs/inference_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/logs/inference_%j.err
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --qos=none
#SBATCH --exclude=node206
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
set -euo pipefail
REPO=/n/fs/similarity/social_sim
source "$REPO/.runtime/oat_env.sh"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
TASK="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/evidence_use"
"$REPO/.runtime/oat_conda/bin/python" "$TASK/run_inference.py" --disclosure "${DISCLOSURE:?set DISCLOSURE}"
"$REPO/.runtime/oat_conda/bin/python" "$TASK/analyze.py"
