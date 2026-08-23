#!/usr/bin/env bash
# Run all three models x all three v11 arms directly on the login node.
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PROJECT_ENV_FILE="/n/fs/similarity/social_sim/exp1_prospective/agent/.env"
RUN_ROOT="$PROJECT_ROOT/data/three_city_c2_v11/full_k1_5_identifiable_structural_choice"
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
    existing_pid="$(<"$PID_DIR/controller.pid")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "local controller already running as PID $existing_pid" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$PID_DIR/controller.pid"

set -a
. "$PROJECT_ENV_FILE"
set +a

if ! python3 eval/validate_three_city_c2_v11_tasks.py; then
    echo "design validation failed; no model processes started" >&2
    exit 1
fi

python3 analysis/live_monitor_three_city_c2_v11.py \
    --watch \
    --interval 60 \
    --draws 300 \
    > "$LOG_DIR/live_monitor.log" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$PID_DIR/live_monitor.pid"

pids=()
labels=()
for model in "${MODELS[@]}"; do
    for arm in "${ARMS[@]}"; do
        label="${model}_${arm}"
        python3 eval/run_three_city_c2_v11_confirmatory.py \
            --model "$model" \
            --arm "$arm" \
            --temperature 0 \
            --max-tokens 512 \
            --timeout 1200 \
            --max-attempts 1 \
            --delay 0.05 \
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
    pid="${pids[$index]}"
    label="${labels[$index]}"
    if wait "$pid"; then
        echo "completed $label pid=$pid"
    else
        child_status=$?
        echo "failed $label pid=$pid status=$child_status" >&2
        status=1
    fi
done

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
python3 analysis/live_monitor_three_city_c2_v11.py --draws 1000

if [[ "$status" -eq 0 ]]; then
    if ! python3 analysis/analyze_three_city_c2_v11_confirmatory.py; then
        status=1
    fi
fi

printf '%s\n' "$status" > "$RUN_ROOT/controller_exit_status.txt"
exit "$status"
