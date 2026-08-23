#!/bin/bash
#SBATCH --job-name=c3_coin_gate
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/gate_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/gate_%j.err
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
"$PY" "$FAMILY/preflight.py"
"$PY" "$FAMILY/canary_audit.py" --ledger "$CAMPAIGN_LEDGER"
