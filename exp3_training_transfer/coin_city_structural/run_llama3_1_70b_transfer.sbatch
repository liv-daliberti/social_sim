#!/bin/bash
# Llama-3.1-70B Coin City structural transfer.
#
# Tests whether the Experiment 3 transfer failure at Llama-3.1-8B is a family
# property or a size threshold. Experiment 2 puts this family's arbitrary-cue
# threshold at 70B against Qwen's 7-14B, so the 8B comparator may simply sit
# below it.
#
# 8x A100-80GB on mltheory, the highest-priority-tier partition available:
# 4 GPUs run one tensor-parallel vLLM actor (320GB for a 141GB model), 4 run the
# ZeRO-3 learner. train.sh assumes one GPU per actor, so this uses train_tp.sh.
#SBATCH --job-name=cc_l70_transfer
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/l70_transfer_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/l70_transfer_%j.err
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --requeue

set -uo pipefail
exec /n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/train_tp.sh
