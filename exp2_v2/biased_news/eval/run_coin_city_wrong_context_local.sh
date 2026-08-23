#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/n/fs/similarity/social_sim/exp2_v2/biased_news"
PYTHON="/usr/bin/python3.9"
RUN_ROOT="$PROJECT_ROOT/data/coin_city_stable_relationship_claude_n250_v4"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
ARM="abc_wrong_context"
C_LEVELS=(0 1 2 3 4)
PLANNED_CALLS_PER_MODEL=1250
HARD_CALL_CAP=5000

# Azure deployments share one project quota, so they run in waves; Gemini sits on
# a different endpoint and key and rides along with the first wave.
WAVE1=("claude-opus-4-8" "claude-opus-5" "gpt-5.6-sol" "gemini-3.6-flash")
WAVE2=("DeepSeek-V4-Pro" "FW-Kimi-K3")

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/responses" "$LOG_DIR" "$PID_DIR"
export PYTHONDONTWRITEBYTECODE=1
if (( PLANNED_CALLS_PER_MODEL > HARD_CALL_CAP )); then
    echo "wrong-context call-cap invariant failed" >&2
    exit 1
fi
if [[ -z "${AZURE_AI_API_KEY:-}" || -z "${GEMINI_API_KEY:-}" ]]; then
    echo "AZURE_AI_API_KEY and GEMINI_API_KEY must both be set" >&2
    exit 1
fi

"$PYTHON" eval/validate_coin_city_stable_relationship_claude_n250.py > \
    "$LOG_DIR/wrong_context_frozen_validator.json" || exit 1

status=0
run_wave() {
    local pids=() labels=()
    for model in "$@"; do
        for c_level in "${C_LEVELS[@]}"; do
            label="${model}_${ARM}_c${c_level}"
            "$PYTHON" eval/run_coin_city_wrong_context.py \
                --model "$model" --prefixes "$c_level" \
                --temperature 0 --max-tokens 512 --timeout 1200 \
                --max-attempts 1 --delay 0.05 \
                > "$LOG_DIR/${label}.log" 2>&1 &
            pid=$!
            pids+=("$pid"); labels+=("$label")
            printf '%s\n' "$pid" > "$PID_DIR/${label}.pid"
            echo "started $label pid=$pid"
        done
    done
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "completed ${labels[$index]}"
        else
            echo "failed ${labels[$index]}" >&2
            status=1
        fi
    done
}

echo "execution_mode=local_login_node host=$(hostname) controller=$$"
run_wave "${WAVE1[@]}"
run_wave "${WAVE2[@]}"

for model in "${WAVE1[@]}" "${WAVE2[@]}"; do
    if ! "$PYTHON" eval/run_coin_city_wrong_context.py \
        --model "$model" --temperature 0 --max-tokens 512 \
        --timeout 1200 --max-attempts 1 --delay 0.05 \
        > "$LOG_DIR/${model}_${ARM}_finalize.log" 2>&1; then
        status=1
    fi
done
printf '%s\n' "$status" > "$RUN_ROOT/wrong_context_controller_exit_status.txt"
exit "$status"
