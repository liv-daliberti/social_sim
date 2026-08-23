#!/bin/bash
#SBATCH --job-name=c3_verify_paper
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/logs/verify_paper_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/logs/verify_paper_%j.err
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
PY="$REPO/.runtime/oat_conda/bin/python"
: "${MECHANISM_LEDGER:?MECHANISM_LEDGER is required}"
: "${POLY_EXTENSION_LEDGER:?POLY_EXTENSION_LEDGER is required}"
: "${COIN_LEDGER:?COIN_LEDGER is required}"
: "${HANDOFF:?HANDOFF is required}"
: "${EXPECTED_VERIFIER_SHA256:?EXPECTED_VERIFIER_SHA256 is required}"
: "${EXPECTED_MECHANISM_LEDGER_SHA256:?EXPECTED_MECHANISM_LEDGER_SHA256 is required}"
: "${EXPECTED_POLY_EXTENSION_LEDGER_SHA256:?EXPECTED_POLY_EXTENSION_LEDGER_SHA256 is required}"
: "${EXPECTED_COIN_LEDGER_SHA256:?EXPECTED_COIN_LEDGER_SHA256 is required}"
: "${EXPECTED_HANDOFF_SHA256:?EXPECTED_HANDOFF_SHA256 is required}"

verify_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label hash mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
}

VERIFIER="$REPO/exp3_training_transfer/verify_paper_handoff.py"
verify_hash "$VERIFIER" "$EXPECTED_VERIFIER_SHA256" "verifier"
verify_hash "$MECHANISM_LEDGER" "$EXPECTED_MECHANISM_LEDGER_SHA256" "mechanism ledger"
verify_hash "$POLY_EXTENSION_LEDGER" "$EXPECTED_POLY_EXTENSION_LEDGER_SHA256" "Polymarket ledger"
verify_hash "$COIN_LEDGER" "$EXPECTED_COIN_LEDGER_SHA256" "Coin City ledger"
verify_hash "$HANDOFF" "$EXPECTED_HANDOFF_SHA256" "handoff"

"$PY" "$VERIFIER" \
  --mechanism-ledger "$MECHANISM_LEDGER" \
  --polymarket-ledger "$POLY_EXTENSION_LEDGER" \
  --coin-ledger "$COIN_LEDGER" \
  --handoff "$HANDOFF"
echo "Both campaigns and both paper PDFs passed the registered handoff verification"
