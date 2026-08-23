#!/bin/bash
#SBATCH --job-name=recovery_eval
#SBATCH --output=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/recovery_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/recovery_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio

# News-response (latent-recovery) eval. Pass MODEL/N via --export; large models
# need a big GPU, e.g.:
#   sbatch --export=ALL,MODEL=qwen2.5:7b,N=400 eval/slurm_recovery.sh
#   sbatch --gres=gpu:a100:1 --mem=64G --export=ALL,MODEL=qwen2.5:72b,N=400 eval/slurm_recovery.sh

set -euo pipefail
ROOT="/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news"
cd "$ROOT"; mkdir -p logs data/recovery

MODEL="${MODEL:-qwen2.5:7b}"
N="${N:-400}"
OLLAMA_BIN="/n/fs/similarity/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/n/fs/similarity/ollama/models"
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
ENDPOINT="http://localhost:${OLLAMA_PORT}/v1"

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  Model $MODEL  N $N  Endpoint $ENDPOINT"

OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
    "$OLLAMA_BIN" serve > /tmp/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1 && { echo "Ollama ready ($i)"; break; }
    sleep 2
done
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" "$OLLAMA_BIN" pull "$MODEL" || true

# Preload into VRAM before the timed run: `ollama run` waits with no client timeout,
# so a slow first load (esp. large models) can't trip the 600s per-call limit. Keep it
# resident for the whole job so subsequent calls hit a warm model.
echo "Warming up $MODEL into VRAM ..."
OLLAMA_HOST="http://localhost:${OLLAMA_PORT}" OLLAMA_KEEP_ALIVE=-1 \
    timeout 1800 "$OLLAMA_BIN" run "$MODEL" "Reply with: ok" \
    && echo "warmup done" || echo "warmup returned nonzero (continuing)"

python3 "eval/${SCRIPT:-run_recovery.py}" \
    --backend ollama --model "$MODEL" --endpoint "$ENDPOINT" \
    --n "$N" --delay 0.03 --max-tokens "${MAXTOK:-160}" --temperature 0.1 "$@"

echo "Done — results in $ROOT/data/recovery/"
