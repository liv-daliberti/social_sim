#!/usr/bin/env bash
#SBATCH --job-name=c2v6_solo
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v6_solo_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v6_solo_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

set -euo pipefail

ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
MODEL="${MODEL:?submit with MODEL set}"

cd "$ROOT"
mkdir -p logs data/three_city_c2_v6_solo/pilot_v6_solo

set -a
. "$ENV_FILE"
set +a

export PYTHONDONTWRITEBYTECODE=1

echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME model=$MODEL"
# --out is REQUIRED here. The runner defaults to the blind arm's directory,
# so omitting it appends solo responses into pilot_v6/responses_*.jsonl while
# the blind jobs are still writing to them.
python3 eval/run_three_city_c2_v6_pilot.py \
    --tasks data/three_city_c2_v6_solo/tasks_c2_v6_solo.jsonl \
    --out data/three_city_c2_v6_solo/pilot_v6_solo \
    --variants 0 \
    --model "$MODEL" \
    --temperature 0 \
    --max-tokens 500 \
    --timeout 1200 \
    --max-attempts 2 \
    --delay 0.1
