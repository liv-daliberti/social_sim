#!/usr/bin/env bash
#SBATCH --job-name=c2v8-full-live
#SBATCH --output=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v8_full_live_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_v2/biased_news/logs/c2v8_full_live_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

set -euo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
RUN_CACHE="$PROJECT_ROOT/data/three_city_c2_v8/full_k1_12/cache"
MPL_CACHE="$RUN_CACHE/live-mpl-${SLURM_JOB_ID}"
RUN_TMP="$RUN_CACHE/live-tmp-${SLURM_JOB_ID}"
cd "$PROJECT_ROOT"
mkdir -p "$MPL_CACHE" "$RUN_TMP"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$MPL_CACHE"
export TMPDIR="$RUN_TMP"

python3 analysis/live_monitor_three_city_c2_v8.py \
    --watch \
    --interval 60 \
    --draws 300
