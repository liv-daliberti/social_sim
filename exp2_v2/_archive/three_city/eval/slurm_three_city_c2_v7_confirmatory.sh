#!/usr/bin/env bash
#SBATCH --job-name=c2v7_confirm
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v7_confirm_%A_%a.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v7_confirm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs
#SBATCH --array=0-2

set -euo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PROJECT_ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_ROOT"
mkdir -p logs data/three_city_c2_v7/confirmatory

set -a
. "$PROJECT_ENV_FILE"
set +a

export PYTHONDONTWRITEBYTECODE=1

echo "job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID model=$MODEL"
python3 eval/validate_three_city_c2_v7_tasks.py
python3 eval/run_three_city_c2_v7_confirmatory.py \
    --model "$MODEL" \
    --temperature 0 \
    --max-tokens 512 \
    --timeout 1200 \
    --max-attempts 1 \
    --delay 0.1
