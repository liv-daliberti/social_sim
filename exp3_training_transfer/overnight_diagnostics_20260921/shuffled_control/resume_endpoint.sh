#!/bin/bash
# Resume the post-training endpoint evaluations for a run whose training and
# final adapter completed but whose evaluation block did not.
#
# Why this exists: train.sh runs training and then, in one process, the
# registered stochastic endpoint, the fresh greedy/stochastic endpoints and
# their scoring. Array task 0 of job 31463557 (disclosed / shuffled_target /
# s47) reached 300/300, wrote step_00301/adapter_model.safetensors intact, and
# was then SIGKILLed during teardown, so every evaluation was lost while the
# 15h48m of training survived. This replays ONLY that evaluation block against
# the saved adapter. Training is not re-run and nothing frozen is edited --
# train.sh is left byte-identical because sibling array tasks are executing it.
#
# Usage: OUT=<existing run dir> INDEX=<roster index> bash resume_endpoint.sh
set -euo pipefail

REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
CONTROL="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control"
FROZEN="$CONTROL/frozen_runtime"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FROZEN:$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

: "${OUT:?set OUT to the existing run directory}"
[ -d "$OUT" ] || { echo "no such run dir: $OUT"; exit 1; }

# Take every run parameter from the frozen roster row, exactly as
# run_extension.py does, so the evaluation matches the one that was lost.
eval "$("$PY" - "$CONTROL" "${INDEX:?set INDEX}" <<'PYEOF'
import json, shlex, sys
from pathlib import Path
row = json.loads((Path(sys.argv[1]) / 'extension_roster.json').read_text())[int(sys.argv[2])]
for k, v in row['environment'].items():
    print(f'export {k}={shlex.quote(str(v))}')
PYEOF
)"
export VLLM_RATIO=0.76   # the same transformation run_extension.py applies
unset VLLM_SLEEP || true

echo "resuming endpoints for $OUT"
echo "  disclosure=$DISCLOSURE arm=$ARM seed=$SEED model=$MODEL_KEY"

# --- registered stochastic endpoint (train.sh lines 136-151) ---
if [ "${STOCHASTIC_N:-5}" -gt 0 ] && [ ! -f "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" ]; then
  ADAPTER=$(find "$OUT" -type d -path '*/saved_models/step_*' | sort | tail -1)
  [ -n "$ADAPTER" ] || { echo "final adapter missing; stochastic endpoint aborted"; exit 3; }
  "$PY" "$FROZEN/evaluate_endpoint.py" \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --data "$DATA/heldout" \
    --template "${PROMPT_TEMPLATE:-biased_news}" \
    --structured-output "$STRUCTURED_OUTPUT" \
    --temperature "${STOCHASTIC_TEMP:-0.7}" \
    --n "${STOCHASTIC_N:-5}" \
    --seed "$(( SEED + 20260814 ))" \
    --max-tokens "$GEN_LEN" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output "$OUT/stochastic_n${STOCHASTIC_N:-5}.json"
fi

# --- scoring of the registered endpoints (train.sh lines 153-161) ---
GREEDY=$(find "$OUT" -type f -path '*/eval_results/*.json' | sort -V | tail -1)
[ -n "$GREEDY" ] || { echo "final greedy evaluation missing"; exit 4; }
"$PY" "$FROZEN/report.py" score "$GREEDY" --model "$MODEL_KEY" \
  --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
if [ "${STOCHASTIC_N:-5}" -gt 0 ]; then
  "$PY" "$FROZEN/report.py" score "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
    --model "$MODEL_KEY" --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
fi
echo "complete: $OUT"

# --- fresh evaluation (train.sh lines 164-180) ---
ADAPTER=$(find "$OUT" -type f -path '*/saved_models/step_00301/adapter_model.safetensors' | head -1)
[ -n "$ADAPTER" ] || { echo "final weights absent"; exit 5; }
ADAPTER=$(dirname "$ADAPTER")
"$PY" "$FROZEN/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" \
  --data "$CONTROL/../evidence_use/data/heldout_${DISCLOSURE}" \
  --template "$PROMPT_TEMPLATE" --structured-output forecast_array \
  --temperature 0 --n 1 --seed "$(( SEED + 20260814 ))" \
  --max-tokens "$GEN_LEN" --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/fresh_greedy.json" \
  --secondary-output "$OUT/fresh_stochastic_n5.json" --secondary-temperature 0.7 --secondary-n 5
for NAME in fresh_greedy fresh_stochastic_n5; do
  "$PY" "$FROZEN/report.py" score "$OUT/$NAME.json" --model "$MODEL_KEY" \
    --disclosure "$DISCLOSURE" --arm "$ARM" --seed "$SEED"
done
echo "fresh_complete: $OUT"
