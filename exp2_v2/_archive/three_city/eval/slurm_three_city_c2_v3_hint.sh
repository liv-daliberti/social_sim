#!/usr/bin/env bash
#SBATCH --job-name=c2v3_hint
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v3_hint_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v3_hint_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

set -euo pipefail

ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
MODEL="${MODEL:?submit with MODEL set}"
# Which background variants to run. 0 = no-context only (the paired comparison
# against the blind arm); 1 2 3 add the orthogonal and two cue conditions so the
# full six-panel diagnostics can be produced for this arm too.
VARIANTS="${VARIANTS:-0}"

cd "$ROOT"
mkdir -p logs data/three_city_c2_v3_hint/pilot_v3_hint

set -a
. "$ENV_FILE"
set +a

export PYTHONDONTWRITEBYTECODE=1

echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME model=$MODEL arm=non_blind_hint variants=$VARIANTS"

# The non-blind hint arm. Task IDs match the blind arm so the same answer key
# scores both and the comparison is exactly paired.
# Extended reasoning is permitted, hence the larger token budget.
python3 eval/run_three_city_c2_v3_pilot.py \
    --model "$MODEL" \
    --tasks data/three_city_c2_v3_hint/tasks_c2_v3_hint.jsonl \
    --out data/three_city_c2_v3_hint/pilot_v3_hint \
    --variants $VARIANTS \
    --temperature 0 \
    --max-tokens 1600 \
    --timeout 1200 \
    --max-attempts 2 \
    --delay 0.1
