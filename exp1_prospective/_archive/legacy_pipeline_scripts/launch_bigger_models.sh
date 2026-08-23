#!/usr/bin/env bash
# Submit bigger-model pipeline for Exp 1.
#
# Step 1: Pull all 6 models (~165 GB) — one job on node302.
# Step 2: Run 3 tier forecast jobs in parallel, each depending on the pull job.
#
# Usage (from exp1_prospective/):
#   bash scripts/launch_bigger_models.sh              # full run, all markets
#   bash scripts/launch_bigger_models.sh --pilot      # 5 markets, k=1, quick sanity check
#
# The --pilot flag submits:
#   sbatch --export=N_MARKETS=5,K_RUNS=1 <tier_script>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PILOT=0
for arg in "$@"; do
    [[ "$arg" == "--pilot" ]] && PILOT=1
done

PILOT_EXPORT=""
if [[ "$PILOT" -eq 1 ]]; then
    PILOT_EXPORT="--export=N_MARKETS=5,K_RUNS=1"
    echo "[pilot mode] 5 markets, k=1 per model"
fi

echo "============================================================"
echo " launch_bigger_models.sh"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Step 1: pull models
PULL_JID=$(sbatch --parsable "$SCRIPT_DIR/slurm_pull_bigger_models.sh")
echo " Pull job:    $PULL_JID  (slurm_pull_bigger_models.sh)"

# Step 2: 3 tier forecast jobs, each waiting for pull to succeed
TIER1_JID=$(sbatch --parsable \
    --dependency=afterok:"$PULL_JID" \
    ${PILOT_EXPORT} \
    "$SCRIPT_DIR/slurm_bigger_forecast_tier1.sh")
echo " Tier 1 job: $TIER1_JID  (qwen2.5:14b + llama3.2:11b)"

TIER2_JID=$(sbatch --parsable \
    --dependency=afterok:"$PULL_JID" \
    ${PILOT_EXPORT} \
    "$SCRIPT_DIR/slurm_bigger_forecast_tier2.sh")
echo " Tier 2 job: $TIER2_JID  (qwen2.5:32b + llama3.1:70b)"

TIER3_JID=$(sbatch --parsable \
    --dependency=afterok:"$PULL_JID" \
    ${PILOT_EXPORT} \
    "$SCRIPT_DIR/slurm_bigger_forecast_tier3.sh")
echo " Tier 3 job: $TIER3_JID  (qwen2.5:72b + llama3.3:70b)"

echo ""
echo " Monitor:  squeue -u \$USER"
echo " Logs:     tail -f logs/pull_bigger_${PULL_JID}.out"
echo "           tail -f logs/tier1_${TIER1_JID}.out"
echo "           tail -f logs/tier2_${TIER2_JID}.out"
echo "           tail -f logs/tier3_${TIER3_JID}.out"
echo "============================================================"
