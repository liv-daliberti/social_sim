#!/bin/bash
# Capacity test: the native-cell recipe at Qwen3-14B. See PROTOCOL.md.
# Mirrors native_cells/run.sbatch; the model and its evaluator differ, and the
# resume chunking is kept because 14B is slower than 8B.
#SBATCH --job-name=cap14b
#SBATCH --account=mltheory
#SBATCH --partition=all
#SBATCH --qos=none
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
REPO=/n/fs/similarity/social_sim
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/capacity_14b"
LEARN="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/learnability"
STRUCTURE="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/structure"
source "$REPO/.runtime/oat_env.sh"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY="$REPO/.runtime/oat_conda/bin/python"
CONFIG="${CONFIG:?A frozen configuration is required}"
OUT="$ROOT/runs/$CONFIG"
ATTEMPT="${ATTEMPT:-1}"

"$PY" - "$ROOT" "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]
gate=json.loads((root/'gate.json').read_text())
assert gate['launch'] is True
assert hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()==gate['config_sha256'][name]
assert hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest()==gate['protocol_sha256']
cfg=json.loads((root/f'{name}.json').read_text())
assert hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest()==cfg['data_sha256'], 'training data changed'
assert 'Qwen3-14B' in cfg['model'], 'this harness runs the 14B capacity arm only'
PY

"$PY" "$LEARN/train_sft.py" --config "$ROOT/$CONFIG.json" --output "$OUT" --max-training-seconds 2400
if [ ! -f "$OUT/complete.json" ]; then
  if [ "$ATTEMPT" -ge 6 ]; then echo "Six chunks exhausted; fixed endpoint not reached"; exit 4; fi
  sbatch --parsable --account=mltheory --partition=all --qos=none \
    --export="ALL,CONFIG=$CONFIG,ATTEMPT=$((ATTEMPT+1))" \
    --output="$ROOT/logs/${CONFIG}_%j.out" --error="$ROOT/logs/${CONFIG}_%j.err" \
    "$ROOT/run.sbatch" >> "$OUT/continuation_jobs.txt"
  exit 0
fi

"$PY" - "$OUT" "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]); name=sys.argv[2]
seed=int(name.rsplit('_s',1)[1])
entry={'key':'qwen3_14b_'+name,'model':'Qwen/Qwen3-14B','arm':name,'seed':seed,
       'adapter':str(out/'adapter'),
       'adapter_weights_sha256':hashlib.sha256((out/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()}
(out/'eval_roster.json').write_text(json.dumps([entry],indent=2)+'\n')
PY

"$PY" "$STRUCTURE/evaluate.py" --model Qwen/Qwen3-14B --roster "$OUT/eval_roster.json" \
  --output-dir "$OUT/component_results"
