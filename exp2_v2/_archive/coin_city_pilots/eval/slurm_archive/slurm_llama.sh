#!/bin/bash
#SBATCH --job-name=biased_llama
#SBATCH --output=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/biased_llama_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/biased_llama_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio

# See slurm_qwen.sh for usage notes.

set -euo pipefail

ROOT="/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news"
cd "$ROOT"
mkdir -p logs data/batch

MODEL="llama3.1:8b"
OLLAMA_BIN="/n/fs/similarity/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/n/fs/similarity/ollama/models"

# Use a port derived from job ID to avoid conflicts if two jobs land on same node
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
ENDPOINT="${OLLAMA_HOST:-http://localhost:${OLLAMA_PORT}/v1}"

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL   Endpoint: $ENDPOINT"
echo "Root: $ROOT"

if [[ "$ENDPOINT" == "http://localhost:${OLLAMA_PORT}/v1" ]]; then
    echo "Starting Ollama on port ${OLLAMA_PORT}..."
    OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
        "$OLLAMA_BIN" serve > /tmp/ollama_${SLURM_JOB_ID}.log 2>&1 &
    OLLAMA_PID=$!
    trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
            echo "Ollama ready (attempt $i)"
            break
        fi
        sleep 2
    done

    OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" "$OLLAMA_BIN" pull "$MODEL" || true
fi

python3 eval/run_batch.py \
    --model    "$MODEL" \
    --endpoint "$ENDPOINT" \
    --n 100 \
    --T 12 \
    --variants V0 V1 V2 \
    --delay 0.05 \
    --max-tokens 150 \
    --temperature 0.1 \
    "$@"

echo "Done — results in $ROOT/data/batch/"
