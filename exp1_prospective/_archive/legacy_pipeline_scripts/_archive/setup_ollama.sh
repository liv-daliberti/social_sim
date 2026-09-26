#!/usr/bin/env bash
# One-time setup: install Ollama binary and pull Qwen/Llama models.
# Run this ONCE from an interactive GPU session on node302 (or any GPU node).
#
# Usage:
#   srun -p gpu --gres=gpu:1 -t 1:00:00 -w node302 --pty bash
#   bash /n/fs/similarity/social_sim/exp1_prospective/scripts/setup_ollama.sh

set -euo pipefail

OLLAMA_BIN_DIR="${OLLAMA_BIN_DIR:-/n/fs/similarity/ollama/bin}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/n/fs/similarity/ollama/models}"
OLLAMA_BIN="$OLLAMA_BIN_DIR/ollama"

echo "============================================================"
echo " Ollama setup for exp1 local model pipeline"
echo " Binary:  $OLLAMA_BIN"
echo " Models:  $OLLAMA_MODELS_DIR"
echo "============================================================"
echo ""

mkdir -p "$OLLAMA_BIN_DIR" "$OLLAMA_MODELS_DIR"

# ── Install Ollama binary (no sudo, install to shared FS) ─────────────────────
if [[ -x "$OLLAMA_BIN" ]]; then
    echo "[ok] Ollama already installed: $($OLLAMA_BIN --version 2>/dev/null || echo '?')"
else
    echo "[1/3] Downloading Ollama binary …"
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        OLLAMA_URL="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64"
    elif [[ "$ARCH" == "aarch64" ]]; then
        OLLAMA_URL="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64"
    else
        echo "[error] Unsupported arch: $ARCH"
        exit 1
    fi
    curl -fsSL "$OLLAMA_URL" -o "$OLLAMA_BIN"
    chmod +x "$OLLAMA_BIN"
    echo "[ok] Ollama installed: $($OLLAMA_BIN --version)"
fi

# ── Pull models ────────────────────────────────────────────────────────────────
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"

# Start temporary Ollama server for pulling
export OLLAMA_HOST="127.0.0.1:11434"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null" EXIT

echo "[2/3] Waiting for Ollama server to start …"
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "[ok] Ollama server is up"
        break
    fi
    sleep 2
done

echo "[3/3] Pulling models …"

echo "  → qwen2.5:7b (~4.7 GB) …"
"$OLLAMA_BIN" pull qwen2.5:7b

echo "  → llama3.1:8b (~4.9 GB) …"
"$OLLAMA_BIN" pull llama3.1:8b

echo ""
echo "[ok] Models available:"
"$OLLAMA_BIN" list

kill "$OLLAMA_PID" 2>/dev/null || true
echo ""
echo "============================================================"
echo " Setup complete."
echo " Run the pipeline with:"
echo "   sbatch scripts/slurm_local_forecast.sh"
echo "============================================================"
