#!/bin/bash
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
repo=/n/fs/similarity/social_sim
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
"$repo/.runtime/oat_conda/bin/python" -u -m exp1_prospective.context_reversal.finalize_robustness --submission "$1"
