#!/bin/bash
#SBATCH --job-name=c3_q14_render
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/q14_finalize_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/q14_finalize_%j.err
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
PY="$REPO/.runtime/oat_conda/bin/python"

: "${EXTENSION_LEDGER:?EXTENSION_LEDGER is required}"
: "${EXPECTED_RENDERER_SHA256:?EXPECTED_RENDERER_SHA256 is required}"
: "${EXPECTED_PROTOCOL_SHA256:?EXPECTED_PROTOCOL_SHA256 is required}"

actual_renderer=$(sha256sum "$FAMILY/render_qwen3_14b_extension.py" | awk '{print $1}')
actual_protocol=$(sha256sum "$FAMILY/QWEN3_14B_SCALE_PROTOCOL.md" | awk '{print $1}')
if [[ "$actual_renderer" != "$EXPECTED_RENDERER_SHA256" ]]; then
  echo "renderer hash mismatch" >&2
  exit 1
fi
if [[ "$actual_protocol" != "$EXPECTED_PROTOCOL_SHA256" ]]; then
  echo "protocol hash mismatch" >&2
  exit 1
fi

"$PY" "$FAMILY/render_qwen3_14b_extension.py" --ledger "$EXTENSION_LEDGER"
(
  cd "$REPO/paper"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)
echo "Qwen3-14B Coin City extension rendered and paper compiled"
