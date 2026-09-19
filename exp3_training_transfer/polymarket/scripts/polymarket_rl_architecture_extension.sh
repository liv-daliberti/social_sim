#!/bin/bash
#SBATCH --job-name=exp4_arch_train
#SBATCH --output=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/arch_train_%j.out
#SBATCH --error=/n/fs/similarity/social_sim/exp3_training_transfer/polymarket/logs/arch_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=72G
#SBATCH --cpus-per-task=8
#SBATCH --partition=all

set -euo pipefail

REPO=/n/fs/similarity/social_sim
POLY="$REPO/exp3_training_transfer/polymarket"
FAMILY="$REPO/exp3_training_transfer/mechanism_family"
DATA="$POLY/data/exp3b_registered"
source "$REPO/.runtime/oat_env.sh"
PY="$REPO/.runtime/oat_conda/bin/python"
export PYTHONPATH="$FAMILY:$REPO/exp3_training_transfer/biased_news${PYTHONPATH:+:$PYTHONPATH}"

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
MODEL="${MODEL:?MODEL is required}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?MODEL_CHECKPOINT is required}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:?PROMPT_TEMPLATE is required}"
SEED="${SEED:?SEED is required}"
RUN_KIND="${RUN_KIND:-train}"
MAX_TRAIN="${MAX_TRAIN:-4800}"
EVAL_STEPS="${EVAL_STEPS:-25}"

case "$MODEL_KEY:$MODEL_CHECKPOINT:$PROMPT_TEMPLATE" in
  qwen3_1_7b:Qwen/Qwen3-1.7B:auto_no_think) ;;
  llama3_2_3b:unsloth/Llama-3.2-3B-Instruct:auto) ;;
  *) echo "unregistered architecture checkpoint/template" >&2; exit 2 ;;
esac
if [ "$RUN_KIND" = train ] && [ "$MAX_TRAIN" -ne 4800 ]; then
  echo "full architecture training requires 4,800 prompt presentations" >&2
  exit 2
fi
if [ "$RUN_KIND" = canary ] && [ "$MAX_TRAIN" -ne 160 ]; then
  echo "architecture canary requires 160 prompt presentations" >&2
  exit 2
fi
if [ "$RUN_KIND" != train ] && [ "$RUN_KIND" != canary ]; then
  echo "invalid RUN_KIND=$RUN_KIND" >&2
  exit 2
fi

OUT="$POLY/reports/scale_${RUN_KIND}_${MODEL_KEY}_market_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-0}"
mkdir -p "$OUT" "$POLY/logs"
cp "$DATA/manifest.json" "$OUT/dataset_manifest.json"
cp "$POLY/data/exp4_scale_registered/manifest.json" "$OUT/scale_holdout_manifest.json"
cp "$POLY/ARCHITECTURE_PROVIDER_SWEEP_PROTOCOL.md" "$OUT/ARCHITECTURE_PROVIDER_SWEEP_PROTOCOL.md"

BATCH=16
ROLLOUT_PER_PROMPT=8
cd "$REPO/.runtime/oat"
"$PY" -c 'from output_contract import FORECAST_ARRAY_GBNF, parse_forecast_array; assert FORECAST_ARRAY_GBNF and callable(parse_forecast_array)'
"$PY" "$FAMILY/run_mechanism_rl.py" \
  --critic_type drgrpo \
  --gpus 1 \
  --num_gpus_per_actor 1 \
  --collocate \
  --lora_rank 32 \
  --lora_alpha 64 \
  --vllm_gpu_ratio "${VLLM_RATIO:-0.38}" \
  --max_model_len 1920 \
  --enable_prefix_caching \
  --gradient-checkpointing \
  --flash-attn \
  --bf16 \
  --zero-stage 2 \
  --ref_offload \
  --beta 0 \
  --learning_rate 0.000001 \
  --lr_scheduler constant \
  --lr_warmup_ratio 0.03 \
  --num_ppo_epochs 1 \
  --seed "$SEED" \
  --oracle_type reward \
  --oracle math \
  --pretrain "$MODEL" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_data "$DATA/hf_train_schedule" \
  --eval_data "$DATA/hf_dev" \
  --train_split train \
  --input_key input \
  --output_key reference \
  --eval_input_key input \
  --max-train "$MAX_TRAIN" \
  --num_prompt_epoch 1 \
  --prompt_max_length 1792 \
  --num_samples "$ROLLOUT_PER_PROMPT" \
  --temperature 1.3 \
  --top_p 1 \
  --generate_max_length 128 \
  --train_batch_size "$BATCH" \
  --train_batch_size_per_device 1 \
  --rollout_batch_size "$BATCH" \
  --rollout_batch_size_per_device "$BATCH" \
  --pi_buffer_maxlen_per_device "$(( BATCH * ROLLOUT_PER_PROMPT ))" \
  --eval_batch_size 128 \
  --eval_steps "$EVAL_STEPS" \
  --eval_temperature 0.7 \
  --eval_top_p 0.8 \
  --eval_top_k 20 \
  --eval_n 5 \
  --eval_generate_max_length 128 \
  --save_steps 999999 \
  --max_save_num 1 \
  --save_path "$OUT" \
  --no-use-wb \
  2>&1 | tee "$OUT/train.log"

if [ "$RUN_KIND" = canary ]; then
  "$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from forecast_scoring import parse_yes_probability

root = Path(sys.argv[1])
evals = sorted(
    root.glob("debug_*/eval_results/*.json"), key=lambda path: int(path.stem)
)
if not evals:
    raise SystemExit("canary wrote no development evaluation")
rows = json.loads(evals[-1].read_text(encoding="utf-8"))
if len(rows) != 512:
    raise SystemExit(f"canary development task count is {len(rows)}, not 512")
parsed_draws = 0
covered_tasks = 0
for row in rows:
    outputs = row.get("output", [])
    if len(outputs) != 5:
        raise SystemExit("canary did not produce five draws per task")
    task_parsed = 0
    for output in outputs:
        probability = parse_yes_probability(output)
        if probability is not None:
            parsed_draws += 1
            task_parsed += 1
    covered_tasks += int(task_parsed > 0)
total_draws = len(rows) * 5
coverage = parsed_draws / total_draws
if coverage < 0.99 or covered_tasks != len(rows):
    raise SystemExit(
        "canary failed amended frozen-parser gate: "
        f"{parsed_draws}/{total_draws} draws, {covered_tasks}/{len(rows)} tasks"
    )
print(
    "canary gate passed: "
    f"{parsed_draws}/{total_draws} draws, {covered_tasks}/{len(rows)} tasks"
)
PY
fi

echo "architecture training: $OUT"
