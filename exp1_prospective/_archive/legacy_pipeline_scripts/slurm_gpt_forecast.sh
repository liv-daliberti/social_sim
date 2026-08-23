#!/usr/bin/env bash
#SBATCH --job-name=exp1_gpt_s1
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/gpt_s1_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/gpt_s1_%j.err
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#
# Stage 1 initial forecasts for GPT-5.4 (Azure AI Foundry).
# Resumes automatically — markets already in the output file are skipped.
# Pure Azure API calls — no GPU needed.
#
# Override via --export:
#   K_RUNS     runs per market (default: 5)
#   N_MARKETS  cap (default: 0 = all)

set -euo pipefail

K_RUNS="${K_RUNS:-5}"
N_MARKETS="${N_MARKETS:-0}"

REPO="/n/fs/similarity/social_sim/exp1_prospective"

mkdir -p "$REPO/logs"

echo "============================================================"
echo " exp1 GPT stage1 — SLURM job $SLURM_JOB_ID"
echo " Node:    $(hostname)"
echo " k=$K_RUNS  n=${N_MARKETS:-all}"
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

N_FLAG=""
[[ "$N_MARKETS" -gt 0 ]] && N_FLAG="--n $N_MARKETS"

echo ""
echo "── GPT-5.4 initial forecasts ────────────────────────────────"
# shellcheck disable=SC2086
python3 agent/run_gpt_forecast.py --k "$K_RUNS" --verbose $N_FLAG

echo ""
echo "============================================================"
echo " Job complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
