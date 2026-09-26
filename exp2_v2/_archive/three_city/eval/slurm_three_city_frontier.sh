#!/usr/bin/env bash
#SBATCH --job-name=three_city
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/three_city_%x_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/three_city_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

# Frontier evaluation for the two-reference-cities → third-city experiment.
#
# Examples:
#   sbatch --export=ALL,MODEL=claude-opus-4-8,N=100 eval/slurm_three_city_frontier.sh
#   sbatch --export=ALL,MODEL=DeepSeek-V4-Pro,N=100 eval/slurm_three_city_frontier.sh
#   sbatch --export=ALL,MODEL=gpt-5.4,N=100 eval/slurm_three_city_frontier.sh

set -euo pipefail

ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
MODEL="${MODEL:-gpt-5.4}"
N="${N:-100}"

cd "$ROOT"
mkdir -p logs data/three_city

# Use the same centrally stored credentials as the existing Exp 1/Exp 2 jobs.
set -a
. "$ENV_FILE"
set +a

export PYTHONDONTWRITEBYTECODE=1

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME"
echo "Model $MODEL  triplets $N  target prefixes 0..5"

python3 eval/run_three_city.py \
    --backend azure \
    --azure-auth \
    --model "$MODEL" \
    --n "$N" \
    --out data/three_city \
    --delay 0.2 \
    --max-tokens 1600 \
    --timeout 1200 \
    --temperature 0.1 \
    --verbose

echo "Done — $MODEL"
