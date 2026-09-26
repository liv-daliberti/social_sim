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
runner="$repo/exp1_prospective/context_reversal/compare_reasoning_modes_v2.py"
"$repo/.runtime/oat_conda/bin/python" -c 'import hashlib,pathlib,sys; assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == sys.argv[2], "Frozen comparison code changed"' "$runner" "$2"
"$repo/.runtime/oat_conda/bin/python" -u -m exp1_prospective.context_reversal.compare_reasoning_modes_v2 --submission "$1"
