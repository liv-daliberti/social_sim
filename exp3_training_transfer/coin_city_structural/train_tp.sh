#!/bin/bash
#SBATCH --job-name=c3_coin
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/train_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural/logs/train_%j.err
#SBATCH --partition=cs
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

set -euo pipefail
REPO=/n/fs/similarity/social_sim
FAMILY="$REPO/exp3_training_transfer/coin_city_structural"
MECHANISM="$REPO/exp3_training_transfer/mechanism_family"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
# This campaign is PyTorch-only.  Prevent Transformers from probing the
# installed TensorFlow/Flax stacks on every cold evaluator start over NFS.
export USE_TF=0
export USE_FLAX=0
# Long ZeRO-3 backward passes can strand several GiB in non-expandable CUDA
# segments even though that memory is not allocated. Let PyTorch grow segments
# in place so a later, longer batch can reuse the reserved capacity.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Put the checked-out OAT source ahead of the environment's installed wheel.
# The vendored tree contains the ZeRO-3 adapter-consolidation fix required by
# these jobs; executing an absolute experiment script otherwise makes Python's
# sys.path[0] the mechanism directory and silently falls back to site-packages.
export PYTHONPATH="$REPO/.runtime/oat:$MECHANISM:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

ARM="${ARM:-causal}"
MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
# node302 is the cluster's A100 host.  A verified local snapshot avoids hours
# of repeated 70B shard reads from NFS.  Until the copy marker exists, retain
# the shared-cache fallback so an interrupted staging process cannot break a job.
LOCAL_LLAMA_HF_HOME=/scratch/od2961_hf_llama31_70b
LOCAL_LLAMA_REV=1605565b47bb9346c5515c34102e054115b4f98b
if [[ "$MODEL" == "meta-llama/Llama-3.1-70B-Instruct" &&
      -f "$LOCAL_LLAMA_HF_HOME/.complete_$LOCAL_LLAMA_REV" ]]; then
  export HF_HOME="$LOCAL_LLAMA_HF_HOME"
  export HUGGINGFACE_HUB_CACHE="$LOCAL_LLAMA_HF_HOME/hub"
  export TRANSFORMERS_CACHE="$LOCAL_LLAMA_HF_HOME/hub"
  echo "using verified node-local Llama snapshot: $LOCAL_LLAMA_REV"
fi
SEED="${SEED:-42}"
DATA="${DATA:-$FAMILY/data/$ARM}"
TAG="${TAG:-${ARM}_${MODEL_KEY}_s${SEED}}"

# A preempted or requeued 70B job must not silently start again from round 0.
# Adapter snapshots are written after every successful actor weight broadcast,
# so select the highest complete snapshot belonging to this Slurm job and reuse
# its original report directory. Explicit recovery arguments always win.
if [[ -z "${OUT_DIR:-}" && -z "${RESUME_ADAPTER_DIR:-}" &&
      "$MODEL_KEY" == "llama3_1_70b" && -n "${SLURM_JOB_ID:-}" ]]; then
  latest_resume_dir=""
  latest_resume_root=""
  latest_resume_step=0
  shopt -s nullglob
  candidate_roots=("$FAMILY/reports/${TAG}_"*"_j${SLURM_JOB_ID}")
  for candidate_root in "${candidate_roots[@]}"; do
    candidate_steps=("$candidate_root"/debug_*/lora/step_*)
    for candidate_dir in "${candidate_steps[@]}"; do
      candidate_name="${candidate_dir##*/step_}"
      [[ "$candidate_name" =~ ^[0-9]+$ ]] || continue
      candidate_step=$((10#$candidate_name))
      [[ -s "$candidate_dir/adapter_model.safetensors" &&
         -s "$candidate_dir/adapter_config.json" ]] || continue
      if (( candidate_step > latest_resume_step )); then
        latest_resume_step="$candidate_step"
        latest_resume_dir="$candidate_dir"
        latest_resume_root="$candidate_root"
      fi
    done
  done
  shopt -u nullglob
  if (( latest_resume_step > 0 )); then
    OUT_DIR="$latest_resume_root"
    RESUME_ADAPTER_DIR="$latest_resume_dir"
    RESUME_STEPS="$latest_resume_step"
    echo "auto-resuming Llama job $SLURM_JOB_ID from step $RESUME_STEPS: $RESUME_ADAPTER_DIR"
  fi
fi

OUT="${OUT_DIR:-$FAMILY/reports/${TAG}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}}"
mkdir -p "$OUT" "$FAMILY/logs"

case "$ARM" in causal|population_prior|structureless) ;; *) echo "bad arm: $ARM"; exit 2;; esac
[ -d "$DATA/train" ] || { echo "missing $DATA/train"; exit 2; }
[ -d "$DATA/heldout" ] || { echo "missing $DATA/heldout"; exit 2; }

GPUS="${GPUS:-2}"
BATCH="${BATCH:-16}"
ROLLOUT_PER_PROMPT="${ROLLOUT_PER_PROMPT:-8}"
[ $(( GPUS % 2 )) -eq 0 ] || { echo "separated actor/learner requires an even GPU count"; exit 2; }
TP="${NUM_GPUS_PER_ACTOR:-1}"
ACTOR_GPUS="$(( GPUS / 2 ))"
[ $(( ACTOR_GPUS % TP )) -eq 0 ] || { echo "actor GPUs not divisible by tensor-parallel size"; exit 2; }
ROLLOUT_WORKERS="$(( ACTOR_GPUS / TP ))"
# oat asserts rollout_batch_size % (learner world_size * per_device) == 0, and its
# world size is the learner GPU count, not the actor count. With one GPU per actor
# the two coincide, which is why the stock script divides by the actor count; under
# tensor parallelism they diverge and the per-device figure must follow the learner.
LEARNER_GPUS="$(( GPUS / 2 ))"
[ $(( BATCH % LEARNER_GPUS )) -eq 0 ] || { echo "batch not divisible by learner GPU count"; exit 2; }
PROMPTS_PER_ACTOR="$(( BATCH / LEARNER_GPUS ))"
MAX_TRAIN="${MAX_TRAIN:-4800}"
EFFECTIVE_MAX_TRAIN="$MAX_TRAIN"
GEN_LEN="${GEN_LEN:-192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
OAT_EVAL_STEPS="${EVAL_STEPS:-50}"
RESUME_ARGS=()
if [[ -n "${RESUME_ADAPTER_DIR:-}" ]]; then
  [[ -d "$RESUME_ADAPTER_DIR" ]] || {
    echo "resume adapter directory missing: $RESUME_ADAPTER_DIR" >&2
    exit 2
  }
  [[ "${RESUME_STEPS:-0}" =~ ^[1-9][0-9]*$ ]] || {
    echo "RESUME_STEPS must be a positive integer with RESUME_ADAPTER_DIR" >&2
    exit 2
  }
  RESUME_ARGS+=(
    --resume_adapter_dir "$RESUME_ADAPTER_DIR"
    --resume_steps "$RESUME_STEPS"
  )
fi
ENDPOINT_LIMIT_ARGS=()
# Feasibility jobs should exercise the entire train/save/reload path without
# spending most of their allocation on four redundant full held-out passes.
# Full experiment tags keep the complete evaluation protocol unchanged.
case "$TAG" in
  canary_*|fit4_*)
    # One global batch reaches backward, optimizer update, LoRA actor sync,
    # final adapter save, and the endpoint reload gate.
    EFFECTIVE_MAX_TRAIN="$BATCH"
    OAT_EVAL_STEPS=-1
    ENDPOINT_LIMIT_ARGS=(--max-prompts 16)
    ;;
esac

echo "Coin City structural transfer: arm=$ARM model=$MODEL seed=$SEED"
echo "data=$DATA output=$OUT max_train=$EFFECTIVE_MAX_TRAIN"

# Per-process Triton cache: see tp_shim/sitecustomize.py. Tensor-parallel vLLM
# workers otherwise race on the shared job-level cache and one dies with ENOTEMPTY.
export PYTHONPATH="$REPO/exp3_training_transfer/coin_city_structural/tp_shim${PYTHONPATH:+:$PYTHONPATH}"

# Tensor-parallel actors run under a Ray executor (see oat/interface.py). Ray
# keeps cluster state in a shared per-node directory, so a job killed mid-run
# leaves a session behind that the next job on that node trips over during
# vllm.LLM init. Give each job its own session root.
export RAY_TMPDIR="${OAT_RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-local}}"
mkdir -p "$RAY_TMPDIR"


cd "$REPO/.runtime/oat"
"$PY" -c 'from pathlib import Path; import oat; p = Path(oat.__file__).resolve(); expected = Path.cwd().resolve(); print(f"OAT source: {p}"); assert expected in p.parents, f"stale OAT import: {p}"'
"$PY" "$MECHANISM/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus "$GPUS" \
  --num_gpus_per_actor "$TP" \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_sync_only \
  --vllm_gpu_ratio "${VLLM_RATIO:-0.38}" \
  --max_model_len "$MAX_MODEL_LEN" \
  --enable_prefix_caching \
  --disable-custom-all-reduce \
  --gradient-checkpointing \
  --gradient-checkpointing-use-reentrant \
  --activation-offloading \
  --flash-attn \
  --bf16 \
  --zero-stage "${ZERO_STAGE:-2}" \
  ${ZERO_STAGE:+$([ "${ZERO_STAGE}" = "3" ] && echo --no-use_fused_lm_head)} \
  --ref_offload \
  --beta 0 \
  --learning_rate "${LR:-0.000001}" \
  --lr_scheduler constant \
  --lr_warmup_ratio "${LR_WARMUP_RATIO:-0.03}" \
  --num_ppo_epochs 1 \
  --seed "$SEED" \
  --oracle_type reward \
  --oracle math \
  --reward_scale_pts 10 \
  --slope_weight "${RESPONSE_W:-0.60}" \
  --slope_scale_g 4 \
  --pretrain "$MODEL" \
  --prompt_template "${PROMPT_TEMPLATE:-biased_news}" \
  --structured_output forecast_array \
  --prompt_data "$DATA/train" \
  --eval_data "$DATA/heldout" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train "$EFFECTIVE_MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length "${PROMPT_MAX_LENGTH:-2304}" \
  --num_samples "$ROLLOUT_PER_PROMPT" \
  --temperature "${TEMP:-1.3}" \
  --top_p 1 \
  --generate_max_length "$GEN_LEN" \
  --train_batch_size "$BATCH" \
  --train_batch_size_per_device 1 \
  --rollout_batch_size "$BATCH" \
  --rollout_batch_size_per_device "$PROMPTS_PER_ACTOR" \
  --pi_buffer_maxlen_per_device "$(( PROMPTS_PER_ACTOR * ROLLOUT_PER_PROMPT ))" \
  --eval_batch_size "${EVAL_BATCH:-120}" \
  --eval_steps "$OAT_EVAL_STEPS" \
  --eval_temperature 0 \
  --eval_n 1 \
  --eval_generate_max_length "$GEN_LEN" \
  --save_steps 999999 \
  "${RESUME_ARGS[@]}" \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee ${RESUME_ADAPTER_DIR:+-a} "$OUT/train.log"

ADAPTER=$(find "$OUT" -type d -path '*/saved_models/step_*' | sort -V | tail -1)
[ -n "$ADAPTER" ] || { echo "final adapter missing"; exit 3; }
"$PY" "$MECHANISM/evaluate_endpoint.py" \
  --model "$MODEL" --adapter "$ADAPTER" --max-lora-rank "${LORA_RANK:-32}" \
  --data "$DATA/heldout" \
  --template "${PROMPT_TEMPLATE:-biased_news}" --structured-output forecast_array \
  --tensor-parallel-size "$TP" --disable-custom-all-reduce \
  --gpu-memory-utilization "${VLLM_RATIO:-0.38}" \
  "${ENDPOINT_LIMIT_ARGS[@]}" \
  --temperature 0 --n 1 --seed "$(( SEED + 20260818 ))" \
  --max-tokens "$GEN_LEN" --max-model-len "$MAX_MODEL_LEN" \
  --output "$OUT/greedy.json" \
  --secondary-output "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
  --secondary-temperature "${STOCHASTIC_TEMP:-0.7}" \
  --secondary-n "${STOCHASTIC_N:-5}"

"$PY" "$FAMILY/report.py" score "$OUT/greedy.json" \
  --model "$MODEL_KEY" --arm "$ARM" --seed "$SEED"
"$PY" "$FAMILY/report.py" score "$OUT/stochastic_n${STOCHASTIC_N:-5}.json" \
  --model "$MODEL_KEY" --arm "$ARM" --seed "$SEED"
echo "complete: $OUT"
