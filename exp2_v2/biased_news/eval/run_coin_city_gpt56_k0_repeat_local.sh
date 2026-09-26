#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PYTHON="$PROJECT_ROOT/.venv_hosted/bin/python"
RUN_ROOT="$PROJECT_ROOT/data/coin_city_stable_relationship_claude_n250_v4"
REPEAT_ROOT="$RUN_ROOT/responses/gpt56_k0_repeat_20260826"
LOG_DIR="$RUN_ROOT/logs/gpt56_k0_repeat_20260826"
PID_DIR="$RUN_ROOT/pids"
MODEL="gpt-5.6-sol"
ARMS=("abc_no_context" "abc_context" "abc_symbol_context")

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$REPEAT_ROOT" "$LOG_DIR" "$PID_DIR"
export PYTHONDONTWRITEBYTECODE=1

controller_file="$PID_DIR/gpt56_k0_repeat_controller.pid"
if [[ -f "$controller_file" ]]; then
    existing="$(<"$controller_file")"
    if [[ "$existing" =~ ^[0-9]+$ ]] && kill -0 "$existing" 2>/dev/null; then
        echo "GPT-5.6 k=0 repeat already running pid=$existing" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$controller_file"

pids=()
labels=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    rm -f "$controller_file"
}
trap cleanup INT TERM EXIT

for arm in "${ARMS[@]}"; do
    "$PYTHON" eval/run_coin_city_gpt56_k0_repeat.py \
        --model "$MODEL" --arm "$arm" --dry-run \
        > "$LOG_DIR/dry_run_${arm}.json" || exit 1
done

for arm in "${ARMS[@]}"; do
    label="${MODEL}_${arm}_repeat"
    "$PYTHON" eval/run_coin_city_gpt56_k0_repeat.py \
        --model "$MODEL" --arm "$arm" --temperature 0 --max-tokens 512 \
        --timeout 1200 --max-attempts 1 --delay 0.05 \
        > "$LOG_DIR/${label}.log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
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

printf '%s\n' "$status" > "$REPEAT_ROOT/controller_exit_status.txt"
trap - INT TERM EXIT
rm -f "$controller_file"
exit "$status"
