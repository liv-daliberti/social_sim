#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_suite() {
  local label="$1"
  local directory="$2"
  shift 2
  printf '\n==> %s\n' "$label"
  (
    cd "$REPO_ROOT/$directory"
    python -m pytest -q "$@"
  )
}

run_suite "Experiment 1 human-materials review" "." \
  exp1_prospective/stage3_materials_annotation/tests
run_suite "Experiment 2 appendix probe and deployed viewer" \
  "exp2_simulated_worlds/biased_news" tests
run_suite "Experiment 2 final Coin City" "exp2_v2/biased_news" tests
run_suite "Experiment 3 Coin City transfer" \
  "exp3_training_transfer/coin_city_structural" tests
run_suite "Experiment 3 structural OOD" \
  "exp3_training_transfer/mechanism_family" tests
run_suite "Experiment 4 historical Polymarket" \
  "exp3_training_transfer/polymarket" tests
run_suite "Deployed Experiment 3 viewer" \
  "exp3_training_transfer/elections" tests
run_suite "Paper layout" "." paper/tests
run_suite "Repository boundaries" "." tests
