#!/bin/bash
#SBATCH --job-name=c3_coin_paper
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/finalize_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/finalize_%j.err
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
EXPECTED_RENDERER_SHA256=${EXPECTED_RENDERER_SHA256:?EXPECTED_RENDERER_SHA256 is required}
ACTUAL_RENDERER_SHA256=$(sha256sum "$FAMILY/make_paper_outputs.py" | awk '{print $1}')
if [[ "$ACTUAL_RENDERER_SHA256" != "$EXPECTED_RENDERER_SHA256" ]]; then
  echo "renderer hash mismatch: expected $EXPECTED_RENDERER_SHA256, found $ACTUAL_RENDERER_SHA256" >&2
  exit 1
fi
"$PY" "$FAMILY/make_paper_outputs.py" --ledger "$CAMPAIGN_LEDGER"
(
  cd "$REPO/paper"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)
echo "Coin City tables rendered and the ICLR paper compiled"
