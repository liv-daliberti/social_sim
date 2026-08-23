#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
RUN_ROOT="$PROJECT_ROOT/data/three_city_c2_v13/full_k1_5_two_regime_structural_choice"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
CACHE_DIR="$RUN_ROOT/cache"
MODELS=("claude-opus-4-8" "DeepSeek-V4-Pro" "gpt-5.4")
ARMS=("c_only" "abc" "abc_structural_clue")

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR" "$CACHE_DIR/tmp" "$CACHE_DIR/matplotlib"
export TMPDIR="$CACHE_DIR/tmp"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
if [[ -f "$PID_DIR/controller.pid" ]]; then
    existing="$(<"$PID_DIR/controller.pid")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "v13 controller already running pid=$existing" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$PID_DIR/controller.pid"
set -a
. "$ENV_FILE"
set +a
python3 eval/validate_three_city_c2_v13_tasks.py || exit 1
python3 analysis/live_exact_structure_three_city_c2_v13.py --watch --interval 60 --draws 300 > "$LOG_DIR/live_monitor.log" 2>&1 &
monitor=$!
printf '%s\n' "$monitor" > "$PID_DIR/live_monitor.pid"
pids=()
labels=()
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        label="${model}_${arm}"
        python3 eval/run_three_city_c2_v13_confirmatory.py \
            --model "$model" --arm "$arm" --temperature 0 --max-tokens 512 \
            --timeout 1200 --max-attempts 1 --delay 0.05 \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        labels+=("$label")
        printf '%s\n' "$pid" > "$PID_DIR/${label}.pid"
        echo "started $label pid=$pid"
    done
done
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "completed ${labels[$index]}"
    else
        echo "failed ${labels[$index]}" >&2
        status=1
    fi
done
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
python3 analysis/live_exact_structure_three_city_c2_v13.py --draws 1000
if [[ "$status" -eq 0 ]]; then
    python3 analysis/analyze_three_city_c2_v13_confirmatory.py || status=1
fi
printf '%s\n' "$status" > "$RUN_ROOT/controller_exit_status.txt"
exit "$status"
