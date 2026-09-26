#!/bin/bash
#SBATCH --job-name=structure_arithmetic
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
source "$REPO/.runtime/oat_env.sh"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/structure/arithmetic_followup"
"$REPO/.runtime/oat_conda/bin/python" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'gate.json').read_text())['launch'] is True
PY
"$REPO/.runtime/oat_conda/bin/python" "$ROOT/evaluate_supplement.py" --model "${MODEL:-Qwen/Qwen3-8B}" --tensor-parallel-size "${TP:-1}"
