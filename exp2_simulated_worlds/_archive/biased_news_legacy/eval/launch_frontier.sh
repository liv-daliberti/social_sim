#!/bin/bash
# Run the 3 frontier (Azure AI Foundry) models for Exp 2 biased-news eval.
# Each model is many API calls (N episodes x 2 bias x 3 variants x up to 12 steps),
# so N defaults small. Same run_batch.py + identical seeded episodes as local models,
# so results land in data/batch/ in one comparable schema.
#
# Requires a key in env (any of these; run_batch falls back across them):
#   export AZURE_AI_API_KEY=...        # used for gpt-5.4
#   export CLAUDE_AZURE_API_KEY=...    # used for claude-opus-4-8
#   export DEEPSEEK_AZURE_API_KEY=...  # used for DeepSeek-V4-Pro
#
# Usage:  bash eval/launch_frontier.sh            # all 3, background, N=20
#         N=50 bash eval/launch_frontier.sh
#         bash eval/launch_frontier.sh --dry-run  # forwarded to run_batch
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/batch

N="${N:-20}"
MODELS=(claude-opus-4-8 gpt-5.4 DeepSeek-V4-Pro)

for m in "${MODELS[@]}"; do
    log="logs/frontier_${m}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching $m  (N=$N)  → $log"
    nohup python3 eval/run_batch.py \
        --backend azure --azure-auth \
        --model "$m" \
        --n "$N" \
        --T 12 \
        --variants V0 V1 V2 \
        --delay 0.2 \
        --max-tokens 150 \
        --temperature 0.1 \
        "$@" > "$log" 2>&1 &
done
echo "Launched ${#MODELS[@]} frontier jobs in background. Tail logs in logs/frontier_*.log"
wait
