#!/usr/bin/env bash
#SBATCH --job-name=c2v8-full
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v8_full_%A_%a.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v8_full_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs
#SBATCH --array=0-8

set -euo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PROJECT_ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
RUN_ROOT="$PROJECT_ROOT/data/three_city_c2_v8/full_k1_12"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
ARMS=("blind" "hint" "strong_hint")
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
ARM_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
MODEL="${MODELS[$MODEL_INDEX]}"
ARM="${ARMS[$ARM_INDEX]}"

cd "$PROJECT_ROOT"
mkdir -p logs "$RUN_ROOT/responses" "$RUN_ROOT/cache/task-${SLURM_ARRAY_TASK_ID}"
export TMPDIR="$RUN_ROOT/cache/task-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$RUN_ROOT/cache/task-${SLURM_ARRAY_TASK_ID}/matplotlib"

set -a
. "$PROJECT_ENV_FILE"
set +a

export PYTHONDONTWRITEBYTECODE=1

echo "job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID model=$MODEL arm=$ARM"
python3 eval/validate_three_city_c2_v8_tasks.py
python3 eval/run_three_city_c2_v8_confirmatory.py \
    --model "$MODEL" \
    --arm "$ARM" \
    --temperature 0 \
    --max-tokens 512 \
    --timeout 1200 \
    --max-attempts 1 \
    --delay 0.05
