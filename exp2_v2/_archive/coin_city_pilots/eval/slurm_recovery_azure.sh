#!/bin/bash
#SBATCH --job-name=probe_frontier
#SBATCH --output=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/probe_frontier_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/logs/probe_frontier_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cs

# Frontier (Azure AI Foundry) recovery probe. No GPU — pure API calls — but must
# run on a compute node, since the Foundry host does NOT resolve from the login node.
#   sbatch --export=ALL,MODEL=gpt-5.4,N=250 eval/slurm_recovery_azure.sh
set -euo pipefail
ROOT="/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news"
cd "$ROOT"; mkdir -p logs data/recovery_probe data/recovery_curve_parity

# keys live in exp1's .env
set -a; . /n/fs/similarity/social_sim/exp1_prospective/agent/.env; set +a

MODEL="${MODEL:-gpt-5.4}"; N="${N:-250}"
echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  Model $MODEL  N $N"
echo "DNS check:"; getent hosts forecasting-agents-resource.services.ai.azure.com \
    && echo "  -> resolves" || echo "  -> NXDOMAIN (will fail)"

python3 "eval/${SCRIPT:-run_recovery_probe.py}" \
    --backend azure --azure-auth --model "$MODEL" \
    --n "$N" --delay 0.2 --max-tokens 256 --temperature 0.1 "$@"
echo "Done — ${SCRIPT:-run_recovery_probe.py}"
