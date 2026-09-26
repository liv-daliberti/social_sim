#!/usr/bin/env bash
#SBATCH --job-name=exp1_frontier_s4
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/frontier_s4_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/frontier_s4_%j.err
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#
# Stage 4 updated forecasts for Claude Opus 4.8 and DeepSeek V4-Pro.
# Pure Azure API calls — no GPU needed.
#
# Override via --export:
#   MODELS   space-separated: "claude deepseek" (default: both)

set -euo pipefail

MODELS="${MODELS:-gpt}"

REPO="/n/fs/similarity/social_sim/exp1_prospective"

mkdir -p "$REPO/logs"

echo "============================================================"
echo " exp1 frontier stage4 — SLURM job $SLURM_JOB_ID"
echo " Node:    $(hostname)"
echo " Models:  $MODELS"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
echo "[ok] Python: $(which python3)  $(python3 --version)"

cd "$REPO"

if [[ -f "agent/.env" ]]; then
    set -o allexport
    source "agent/.env"
    set +o allexport
fi

for MODEL in $MODELS; do
    echo ""
    echo "── $MODEL stage 4 ──────────────────────────────────────────"
    case "$MODEL" in
        claude)
            python3 agent/claude_updated_forecast.py --verbose
            ;;
        deepseek)
            python3 agent/deepseek_updated_forecast.py --verbose
            ;;
        gpt)
            python3 agent/gpt_updated_forecast.py --verbose
            ;;
        *)
            echo "[warn] unknown model: $MODEL — skipping"
            ;;
    esac
done

echo ""
echo "============================================================"
echo " Job complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
