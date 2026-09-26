#!/usr/bin/env bash
# Start all viewer sub-servers and the hub.
# Usage: ./start_viewer.sh
# Then open: http://localhost:5050
#
# Tabs:
#   Exp 1 — Prospective : exp1_prospective/viewer/app.py            (5051)
#   Exp 2 — Simulations : exp2_simulated_worlds/biased_news viewer  (5052)  Simulator + Results
#   Exp 3 — Transfer    : exp3_training_transfer/elections/server.py (5053)  static DAG + GRPO training

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$EXP1_PID" "$EXP2_PID" "$EXP3_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "Starting Exp 1 viewer on port 5051..."
python "$ROOT/exp1_prospective/viewer/app.py" --port 5051 &
EXP1_PID=$!

echo "Starting Exp 2 (biased news: simulator + results) on port 5052..."
python "$ROOT/exp2_simulated_worlds/biased_news/viewer/app.py" --port 5052 &
EXP2_PID=$!

echo "Starting Exp 3 (elections: static DAG + training) on port 5053..."
PORT=5053 python "$ROOT/exp3_training_transfer/elections/server.py" &
EXP3_PID=$!

# Give sub-servers a moment to bind
sleep 1

echo ""
echo "  Hub → http://localhost:5050"
echo "  Ctrl-C to stop everything."
echo ""

python "$ROOT/viewer/server.py"
