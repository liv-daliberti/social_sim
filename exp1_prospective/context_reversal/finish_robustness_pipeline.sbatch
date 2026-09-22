#!/bin/bash
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -euo pipefail
repo=/n/fs/similarity/social_sim
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Usage: CONFIG CONFIG_SHA256 [--execute|--dry-run]; default only runs preflight.
"$repo/.runtime/oat_conda/bin/python" -u -m exp1_prospective.context_reversal.finish_robustness_pipeline --config "$1" --expected-config-sha256 "$2" "${3:---dry-run}"
