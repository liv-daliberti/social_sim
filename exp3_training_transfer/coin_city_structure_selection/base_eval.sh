#!/bin/bash
#SBATCH --job-name=sel_base
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/selbase_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/selbase_%j.err
#SBATCH --partition=cs
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Untrained reference endpoint for protocol `coin_city_structure_selection_v1`.
#
# The pilot's primary endpoint is the cue-following index D for the trained
# arms *against the untrained base*. That base did not exist when the pilot was
# submitted -- an omission in the protocol, corrected here.
#
# Distinct from the parent's base_eval.sh rather than a copy of it: this one
# points at the structure-selection held-out set (2,880 tasks, twice the
# parent's, hence the 6h limit against the parent's 2h) and writes to a
# `selbase_` prefix so no parent glob, all of which are anchored at
# `base_{model}_j*`, can match it.

set -euo pipefail
REPO=/n/fs/similarity/social_sim
SELECTION="$REPO/exp3_training_transfer/coin_city_structure_selection"
PARENT="$REPO/exp3_training_transfer/coin_city_structural"
MECHANISM="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
MODEL="${MODEL:?MODEL is required}"
TEMPLATE="${PROMPT_TEMPLATE:-auto_no_think}"
GEN_LEN="${GEN_LEN:-192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"

# Both arms ship byte-identical held-out sets; either is the evaluation universe.
DATA="${DATA:-$SELECTION/data/causal/heldout}"
OUT="$PARENT/reports/selbase_${MODEL_KEY}_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$PARENT/logs"

[ -f "$DATA/dataset_dict.json" ] || { echo "not an Arrow dataset: $DATA"; exit 2; }

echo "structure-selection base endpoint: model=$MODEL data=$DATA out=$OUT"

# Decoder settings identical to the trained arms, so base and trained endpoints
# are comparable: greedy monitor plus the registered five-draw stochastic.
"$PY" "$MECHANISM/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature 0 --n 1 --seed 20260818 \
  --structured-output forecast_array --max-tokens "$GEN_LEN" \
  --max-model-len "$MAX_MODEL_LEN" --output "$OUT/greedy.json"
"$PY" "$MECHANISM/evaluate_endpoint.py" --model "$MODEL" --data "$DATA" \
  --template "$TEMPLATE" --temperature 0.7 --n 5 --seed 20260819 \
  --structured-output forecast_array --max-tokens "$GEN_LEN" \
  --max-model-len "$MAX_MODEL_LEN" --output "$OUT/stochastic_n5.json"

"$PY" "$PARENT/report.py" score "$OUT/greedy.json" --model "$MODEL_KEY" --arm base --seed 0
"$PY" "$PARENT/report.py" score "$OUT/stochastic_n5.json" --model "$MODEL_KEY" --arm base --seed 0
echo "complete: $OUT"
