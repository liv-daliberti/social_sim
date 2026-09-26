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
"$PY" - "$ROOT" "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]; gate=json.loads((root/'gate.json').read_text())
assert gate['launch_forecast_sft'] is True
assert hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()==gate['config_sha256'][name]
PY
"$PY" "$ROOT/train_sft.py" --config "$ROOT/$CONFIG.json" --output "$OUT" --max-training-seconds 2400
if [ ! -f "$OUT/complete.json" ]; then
  if [ "$ATTEMPT" -ge 6 ]; then echo "Six chunks exhausted; fixed endpoint not yet reached"; exit 4; fi
  sbatch --parsable --account=mltheory --partition=all --qos=none --export="ALL,CONFIG=$CONFIG,ATTEMPT=$((ATTEMPT+1))" \
    --output="$ROOT/logs/${CONFIG}_%j.out" --error="$ROOT/logs/${CONFIG}_%j.err" "$ROOT/run.sbatch" >> "$OUT/continuation_jobs.txt"
  exit 0
fi
"$PY" - "$OUT" "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]); name=sys.argv[2]
entry={'key':'qwen3_8b_sft_'+name+'_s42','model':'Qwen/Qwen3-8B','arm':name,'seed':42,'adapter':str(out/'adapter'),'adapter_weights_sha256':hashlib.sha256((out/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()}
(out/'eval_roster.json').write_text(json.dumps([entry],indent=2)+'\n')
PY
sbatch --parsable --account=mltheory --partition=all --qos=none --export="ALL,CONFIG=$CONFIG" \
 --output="$ROOT/logs/eval_${CONFIG}_%j.out" --error="$ROOT/logs/eval_${CONFIG}_%j.err" "$ROOT/evaluate.sbatch" >> "$OUT/evaluation_jobs.txt"
