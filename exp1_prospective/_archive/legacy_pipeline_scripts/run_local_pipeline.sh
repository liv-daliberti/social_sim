#!/usr/bin/env bash
# Step 2b end-to-end pipeline for local small models (Qwen2.5-7B / Llama3.1-8B).
# Run from the exp1_prospective/ directory.
#
# Prerequisites:
#   ollama pull qwen2.5:7b
#   ollama pull llama3.1:8b
#   pip install tavily-python duckduckgo-search
#   export TAVILY_API_KEY=<key>       # optional; DDG used if absent
#
# Usage:
#   bash scripts/run_local_pipeline.sh                  # full run, both models
#   bash scripts/run_local_pipeline.sh --n 5 --k 1      # pilot: 5 markets, 1 run
#   bash scripts/run_local_pipeline.sh --model qwen2.5:7b

set -euo pipefail

cd "$(dirname "$0")/.."

# ── parse args ─────────────────────────────────────────────────────────────────
MODELS=("qwen2.5:7b" "llama3.1:8b")
N_MARKETS=0
K_RUNS=3
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODELS=("$2");     shift 2 ;;
        --n)        N_MARKETS="$2";    shift 2 ;;
        --k)        K_RUNS="$2";       shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift  ;;
    esac
done

N_FLAG=""
[ "$N_MARKETS" -gt 0 ] && N_FLAG="--n $N_MARKETS"

echo "============================================================"
echo " Local model pipeline: Step 2b"
echo " Models:   ${MODELS[*]}"
echo " Markets:  ${N_MARKETS:-all}  k=${K_RUNS}"
echo "============================================================"
echo ""

# ── confirm Ollama is running ──────────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[ERROR] Ollama does not appear to be running."
    echo "  Start it with: ollama serve"
    exit 1
fi

# ── Step 2b: initial forecasts ────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    echo "── Step 2b: $MODEL ──────────────────────────────────────"
    # shellcheck disable=SC2086
    python agent/run_local_forecast.py \
        --model "$MODEL" \
        --k "$K_RUNS" \
        ${N_FLAG} \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo ""
done

# ── Step 4: counterfactual updates ────────────────────────────────────────────
if ls data/counterfactuals/counterfactuals_*.jsonl 1>/dev/null 2>&1; then
    for MODEL in "${MODELS[@]}"; do
        echo "── Step 4 (updated forecasts): $MODEL ──────────────────"
        # shellcheck disable=SC2086
        python agent/local_updated_forecast.py \
            --model "$MODEL" \
            ${N_FLAG} \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        echo ""
    done
else
    echo "── Step 4 skipped: no counterfactuals file found."
    echo "   Run agent/build_counterfactuals.py first, then re-run this script."
fi

echo "============================================================"
echo " Done."
echo "============================================================"
