#!/usr/bin/env bash
#SBATCH --job-name=exp1_ollama_setup
#SBATCH --output=/n/fs/similarity/social_sim/exp1_prospective/logs/setup_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp1_prospective/logs/setup_%j.err
#SBATCH --partition=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodelist=node302
#
# One-time: install Ollama binary to shared FS and pull qwen2.5:7b + llama3.1:8b.
# Submit with: sbatch scripts/slurm_setup_ollama.sh
# Takes ~15-25 min depending on network speed.

set -euo pipefail

# Ollama tarball extracts to  bin/ollama + lib/ollama/  relative to OLLAMA_BASE
OLLAMA_BASE="/n/fs/similarity/ollama"
OLLAMA_BIN_DIR="$OLLAMA_BASE/bin"
OLLAMA_MODELS_DIR="$OLLAMA_BASE/models"
OLLAMA_BIN="$OLLAMA_BIN_DIR/ollama"
OLLAMA_HOST="127.0.0.1:11434"

export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST

mkdir -p "$OLLAMA_BASE" "$OLLAMA_BIN_DIR" "$OLLAMA_MODELS_DIR" \
         /n/fs/similarity/social_sim/exp1_prospective/logs

echo "============================================================"
echo " Ollama setup — SLURM job $SLURM_JOB_ID on $(hostname)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── GPU check ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[warn] no nvidia-smi"

# ── Install Ollama binary ──────────────────────────────────────────────────────
# Ollama ships as tar.zst; use Python zstandard since zstd binary is not on PATH.
# Tarball layout: bin/ollama  lib/ollama/...  — extract to OLLAMA_BASE so paths land right.
if [[ -x "$OLLAMA_BIN" ]]; then
    echo "[ok] Ollama already installed: $("$OLLAMA_BIN" --version 2>/dev/null || echo '?')"
else
    echo "[1/4] Downloading and extracting Ollama v0.30.7 …"
    export _OL_URL="https://github.com/ollama/ollama/releases/download/v0.30.7/ollama-linux-amd64.tar.zst"
    export _OL_DEST="$OLLAMA_BASE"
    python3 -c "
import os, urllib.request, tarfile, zstandard
url  = os.environ['_OL_URL']
dest = os.environ['_OL_DEST']
zst  = '/tmp/ollama_setup.tar.zst'
print(f'  Downloading {url} …', flush=True)
urllib.request.urlretrieve(url, zst)
print('  Extracting …', flush=True)
with open(zst, 'rb') as fh:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as tf:
            tf.extractall(dest)
print(f'  Done → {dest}', flush=True)
"
    chmod +x "$OLLAMA_BIN"
    echo "[ok] $("$OLLAMA_BIN" --version)"
fi

# ── Start Ollama server ────────────────────────────────────────────────────────
echo "[2/4] Starting Ollama server (host=$OLLAMA_HOST) …"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null' EXIT

for i in $(seq 1 20); do
    if curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[ok] Server up"
        break
    fi
    sleep 3
done

if ! curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "[error] Ollama server did not start"
    exit 1
fi

# ── Pull models ────────────────────────────────────────────────────────────────
echo "[3/4] Pulling qwen2.5:7b (~4.7 GB) …"
"$OLLAMA_BIN" pull qwen2.5:7b

echo "[4/4] Pulling llama3.1:8b (~4.9 GB) …"
"$OLLAMA_BIN" pull llama3.1:8b

echo ""
echo "[ok] Models on disk:"
"$OLLAMA_BIN" list

kill "$OLLAMA_PID" 2>/dev/null || true
echo ""
echo "============================================================"
echo " Setup complete — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Submit the forecast job with:"
echo "   sbatch --export=N_MARKETS=5,K_RUNS=1,NO_THIRD=1 \\"
echo "     /n/fs/similarity/social_sim/exp1_prospective/scripts/slurm_local_forecast.sh"
echo "============================================================"
