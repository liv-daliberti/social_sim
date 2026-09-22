#!/bin/bash
#SBATCH --job-name=structure_sft
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --qos=none
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=node206
set -euo pipefail
REPO=/n/fs/similarity/social_sim
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/learnability"
STRUCTURE="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/structure"
source "$REPO/.runtime/oat_env.sh"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY="$REPO/.runtime/oat_conda/bin/python"
CONFIG="${CONFIG:?A frozen configuration is required}"
OUT="$ROOT/runs/${CONFIG}_s42"
ATTEMPT="${ATTEMPT:-1}"
"$PY" "$STRUCTURE/evaluate.py" --model Qwen/Qwen3-8B --roster "$OUT/eval_roster.json" --output-dir "$OUT/component_results"
