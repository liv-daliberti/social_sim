#!/bin/bash
# Submit the full local-model roster for Exp 2 biased-news eval (one slurm job each).
# Usage:  bash eval/launch_local.sh            # submit all 7
#         N=50 bash eval/launch_local.sh       # smaller run per model
#         bash eval/launch_local.sh --dry-run  # forwarded to run_batch (prints prompt)
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
    qwen2.5:7b
    qwen2.5:14b
    qwen2.5:32b
    qwen2.5:72b
    llama3.1:8b
    llama3.1:70b
    llama3.3:70b
)
N="${N:-100}"

for m in "${MODELS[@]}"; do
    echo "Submitting $m  (N=$N)"
    sbatch --job-name="be_${m//:/-}" \
           --export=ALL,MODEL="$m",N="$N" \
           eval/slurm_eval.sh "$@"
done
echo "Submitted ${#MODELS[@]} jobs. Track with:  squeue -u $USER"
