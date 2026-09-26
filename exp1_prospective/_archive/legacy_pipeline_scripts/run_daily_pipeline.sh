#!/bin/bash
# Daily pipeline: fetch fresh markets → select → diversify → enrich with CLOB prices + volume.
# Designed to be run once per day via cron.
#
# Cron example (runs at 14:00 UTC daily):
#   0 14 * * * bash /n/fs/similarity/social_sim/exp1_prospective/scripts/run_daily_pipeline.sh >> /n/fs/similarity/social_sim/exp1_prospective/logs/daily_$(date +\%Y-\%m-\%d).log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

DATE=$(date -u +%Y-%m-%d)
echo "============================================================"
echo " exp1_prospective daily pipeline  —  $DATE"
echo "============================================================"

# ── Step 1: fetch all active Polymarket markets from Gamma API ─────────────────
echo ""
echo "[1/4] Fetching all active markets from Gamma API..."
python3 fetch_markets/fetch_markets.py

# ── Step 2: filter to experiment subset ───────────────────────────────────────
echo ""
echo "[2/4] Selecting experiment subset..."
python3 fetch_markets/select_markets.py

# ── Step 3: diversify ──────────────────────────────────────────────────────────
echo ""
echo "[3/4] Diversifying selected markets..."
python3 fetch_markets/diversify_markets.py

# ── Step 4: enrich with CLOB prices + daily volume ────────────────────────────
echo ""
echo "[4/4] Fetching CLOB price history and today's trade volume..."
python3 fetch_markets/fetch_daily_clob.py --input "data/selected_markets/diverse_${DATE}.jsonl"

echo ""
echo "============================================================"
echo " Pipeline complete  —  $DATE"
echo "============================================================"
