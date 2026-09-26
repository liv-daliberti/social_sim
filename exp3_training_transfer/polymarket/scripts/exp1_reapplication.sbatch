#!/bin/bash
#SBATCH --job-name=exp3b_exp1
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/exp1_reapplication_%A_%a.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/exp1_reapplication_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
REPORT_DIR="$POLY/reports/exp3b_exp1_reapplication_j${SLURM_ARRAY_JOB_ID}"
mkdir -p "$REPORT_DIR"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    NAME=base
    ADAPTER=
    ;;
  1)
    NAME=seed_42
    ADAPTER="$POLY/reports/train_market_20260813_031628_j30505537/debug_0813T03:17:37/saved_models/step_00301"
    ;;
  2)
    NAME=seed_43
    ADAPTER="$POLY/reports/train_market_20260813_035132_j30505538/debug_0813T03:52:00/saved_models/step_00301"
    ;;
  3)
    NAME=seed_44
    ADAPTER="$POLY/reports/train_market_20260813_035633_j30505539/debug_0813T03:56:57/saved_models/step_00301"
    ;;
  *)
    echo "unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

ARGS=(
  --name "$NAME"
  --output "$REPORT_DIR/${NAME}.jsonl"
  --max-tokens 1500
  --max-model-len 6144
)
if [[ -n "$ADAPTER" ]]; then
  ARGS+=(--adapter "$ADAPTER")
fi

"$PY" "$POLY/scripts/evaluate_exp1_reapplication.py" "${ARGS[@]}"
