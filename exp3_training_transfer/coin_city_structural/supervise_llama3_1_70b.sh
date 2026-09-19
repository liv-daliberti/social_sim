#!/bin/bash
# Keep a 4xA100 Llama-3.1-70B run moving after a learner-only failure.
#
# Launchpad can leave the actor alive after the ZeRO learner dies. Slurm then
# reports RUNNING forever while the GPUs are idle. This wrapper watches the
# durable adapter snapshots and train log, terminates only the silent attempt,
# and resumes from the highest complete adapter within the same allocation.

set -uo pipefail

REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
TRAIN_SCRIPT="$FAMILY/train_tp.sh"

ARM="${ARM:-causal}"
MODEL_KEY="${MODEL_KEY:-llama3_1_70b}"
SEED="${SEED:-42}"
TAG="${TAG:-${ARM}_${MODEL_KEY}_s${SEED}}"
TARGET_STEPS="${TARGET_STEPS:-300}"
STALE_SECONDS="${LLAMA_WATCHDOG_STALE_SECONDS:-2400}"
MAX_ATTEMPTS="${LLAMA_WATCHDOG_MAX_ATTEMPTS:-20}"

# A 70B LoRA restore/broadcast can take longer than vLLM's 60-second default
# engine-iteration watchdog. Keep the watchdog comfortably above that restore
# latency so the first rollout after a resume does not kill a healthy engine.
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"

if [[ "$MODEL_KEY" != "llama3_1_70b" ]]; then
  echo "supervisor is restricted to MODEL_KEY=llama3_1_70b" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "supervisor requires a Slurm allocation" >&2
  exit 2
fi

find_latest_snapshot() {
  local best_dir="" best_step=0 candidate_dir candidate_name candidate_step
  local candidate_steps=()
  shopt -s nullglob
  candidate_steps=("$OUT_DIR"/debug_*/lora/step_*)
  shopt -u nullglob
  for candidate_dir in "${candidate_steps[@]}"; do
    candidate_name="${candidate_dir##*/step_}"
    [[ "$candidate_name" =~ ^[0-9]+$ ]] || continue
    candidate_step=$((10#$candidate_name))
    [[ -s "$candidate_dir/adapter_model.safetensors" &&
       -s "$candidate_dir/adapter_config.json" ]] || continue
    if (( candidate_step > best_step )); then
      best_step="$candidate_step"
      best_dir="$candidate_dir"
    fi
  done
  printf '%s\t%s\n' "$best_step" "$best_dir"
}

if [[ -z "${OUT_DIR:-}" ]]; then
  shopt -s nullglob
  existing_roots=("$FAMILY/reports/${TAG}_"*"_j${SLURM_JOB_ID}")
  shopt -u nullglob
  if (( ${#existing_roots[@]} > 0 )); then
    OUT_DIR="${existing_roots[0]}"
  else
    OUT_DIR="$FAMILY/reports/${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}"
  fi
fi
export OUT_DIR
mkdir -p "$OUT_DIR"

echo "Llama supervisor: job=$SLURM_JOB_ID arm=$ARM output=$OUT_DIR"
echo "watchdog: target_steps=$TARGET_STEPS stale_seconds=$STALE_SECONDS"

for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
  IFS=$'\t' read -r resume_step resume_dir < <(find_latest_snapshot)
  if (( resume_step >= TARGET_STEPS )) &&
     [[ -s "$OUT_DIR/greedy.json" &&
        -s "$OUT_DIR/stochastic_n${STOCHASTIC_N:-5}.json" ]]; then
    echo "supervisor_complete step=$resume_step output=$OUT_DIR (pre-existing artifacts)"
    exit 0
  elif (( resume_step >= TARGET_STEPS )); then
    echo "training checkpoint target reached at step $resume_step; resuming endpoint evaluation"
  fi

  if (( resume_step > 0 )); then
    export RESUME_ADAPTER_DIR="$resume_dir"
    export RESUME_STEPS="$resume_step"
    export LR_WARMUP_RATIO=0
  else
    unset RESUME_ADAPTER_DIR RESUME_STEPS
  fi
  export OAT_RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_supervised_${resume_step}_${attempt}"

  echo "attempt=$attempt resume_step=$resume_step started=$(date --iso-8601=seconds)"
  setsid "$TRAIN_SCRIPT" &
  child=$!
  attempt_start=$(date +%s)
  last_step="$resume_step"
  last_activity="$attempt_start"
  stale=0

  while kill -0 "$child" 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    IFS=$'\t' read -r current_step current_dir < <(find_latest_snapshot)
    if (( current_step > last_step )); then
      last_step="$current_step"
      last_activity="$now"
      echo "checkpoint_progress step=$last_step at=$(date --iso-8601=seconds)"
    fi

    if [[ -f "$OUT_DIR/train.log" ]]; then
      log_mtime=$(stat -c %Y "$OUT_DIR/train.log" 2>/dev/null || echo 0)
      if (( log_mtime > last_activity )); then
        last_activity="$log_mtime"
      fi
    fi

    # After training writes a final saved model, endpoint evaluation can be
    # legitimately quiet for longer than a training round.
    final_adapter=$(find "$OUT_DIR" -type d -path '*/saved_models/step_*' -print -quit 2>/dev/null)
    if [[ -z "$final_adapter" &&
          $(( now - last_activity )) -ge "$STALE_SECONDS" &&
          $(( now - attempt_start )) -ge "$STALE_SECONDS" ]]; then
      echo "watchdog_stale attempt=$attempt step=$last_step silent_seconds=$(( now - last_activity ))"
      stale=1
      kill -TERM -- "-$child" 2>/dev/null || true
      for _ in {1..30}; do
        kill -0 "$child" 2>/dev/null || break
        sleep 1
      done
      kill -KILL -- "-$child" 2>/dev/null || true
      break
    fi
  done

  wait "$child"
  rc=$?
  IFS=$'\t' read -r final_step final_dir < <(find_latest_snapshot)
  if (( rc == 0 )) && [[ -s "$OUT_DIR/greedy.json" &&
                         -s "$OUT_DIR/stochastic_n${STOCHASTIC_N:-5}.json" ]]; then
    echo "supervisor_complete step=$final_step output=$OUT_DIR"
    exit 0
  fi

  echo "attempt_exit attempt=$attempt rc=$rc stale=$stale latest_step=$final_step"
  sleep 15
done

echo "supervisor exhausted $MAX_ATTEMPTS attempts" >&2
exit 1
