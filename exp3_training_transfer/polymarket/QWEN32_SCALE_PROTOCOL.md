# Experiment 4 Qwen3-32B prospective scale extension

Protocol version: `exp4_qwen32_scale_v1`

## Registration and relation to the parent experiment

This extension was registered on August 25, 2026 before any model was run on
the sealed Experiment 4 scale holdout.  The already-submitted Qwen3-4B,
Qwen3-8B, and Qwen3-14B locked-test jobs (`30870546`, `30870547`, and
`30870548`) and their finalizer (`30870549`) were placed in a user hold before
the final Qwen3-14B training seed completed.  Scheduler accounting showed that
none of the evaluation jobs had started.  They remain held until all three
Qwen3-32B training seeds complete successfully.

The parent `exp4_qwen_scale_v1` protocol, its 4B/8B/14B roster, and its primary
Qwen3-14B estimand are unchanged.  This document prospectively adds a separate
Qwen3-32B extension.  The registration ledger freezes the hashes of this
protocol, the parent holdout, and every training, evaluation, rendering, and
launch script before the Qwen3-32B canary is submitted.

## Question and model

Does the base-relative forecasting gain replicate at Qwen3-32B, and does the
trained dense Qwen3 series continue to improve from 8B to 14B to 32B?

The only added checkpoint is the dense `Qwen/Qwen3-32B`.  Qwen3-30B-A3B is
excluded because its mixture-of-experts architecture would confound a dense
parameter-scaling comparison.  Qwen3-14B remains in the parent roster and is
not replaced.

## Training specification

Qwen3-32B is trained at seeds 42, 43, and 44.  All scientific settings remain
identical to the parent Qwen3-14B protocol:

- rank-32 LoRA with alpha 64; Dr. GRPO; beta 0;
- learning rate `1e-6`, constant schedule, and 3% warmup;
- the frozen deterministic 4,800-presentation training schedule;
- 16 prompts per round, eight stochastic rollouts per prompt, and 300 rounds;
- rollout temperature 1.3 and top-p 1;
- pure reward `1 - (p - y)^2`;
- bf16, gradient checkpointing, FlashAttention, and reference offload; and
- the same 512-task development split evaluated every 25 rounds with five
  stochastic non-thinking draws per task.

Only memory-distribution settings change.  Each job uses four A6000 GPUs: one
two-GPU tensor-parallel vLLM rollout worker and a two-rank learner using ZeRO
stage 3.  The per-learner rollout batch is eight and per-learner policy buffer
capacity is 64, preserving the registered global batches of 16 prompts and 128
rollout transitions.  These changes do not alter examples, optimizer settings,
sampling, rewards, or the number of parameter updates.

## Feasibility gate

A ten-round seed-42 canary is submitted first.  Full jobs are released only if
the canary completes, saves a valid adapter, has finite reward metrics, and all
2,560 development draws parse.  This is a feasibility-only gate: no accuracy,
Brier, reward, or improvement threshold is used.  Canary output is never a
paper result and no hyperparameter may be tuned from its performance.

## Locked stochastic evaluation

After all three Qwen3-32B seeds complete, the held parent evaluations are
released and Qwen3-32B is evaluated on the same still-sealed holdout.  Every
base and trained endpoint uses five non-thinking stochastic draws per task,
temperature 0.7, top-p 0.8, top-k 20, the parent's fixed sampling seed, and at
most 128 new tokens.  Probabilities are averaged within a task only when all
five draws parse; otherwise that endpoint/task receives Brier loss one.

The Qwen3-32B evaluator differs from the parent evaluator only by using
two-way tensor parallelism so the dense checkpoint fits in A6000 memory.  Raw
draws and both draw-level and complete-task parse coverage are retained.

## Estimands and reporting

The extension's primary contrast is the three-seed mean Qwen3-32B trained
Brier loss minus its untrained base on the sealed holdout.  Secondary contrasts
are:

- trained Qwen3-32B minus trained Qwen3-14B;
- trained Qwen3-32B minus trained Qwen3-8B;
- trained Qwen3-32B minus the latest genuine pre-cutoff market price; and
- trained Qwen3-32B minus the train-only Platt market baseline.

Intervals use the parent's registered 5,000-repetition connected-family
bootstrap.  Negative deltas favor the first named model.  Superiority requires
the upper 95% confidence bound below zero.  All three seeds, the complete 32B
result, and every registered contrast are reported regardless of direction.

## Fail-closed execution

Any model-cache, protocol, source-code, parent-ledger, holdout, adapter, seed,
task, draw, or dependency mismatch aborts the handoff.  If the canary or any
full Qwen3-32B seed fails, the parent locked-test jobs remain held and the
holdout remains unopened until an explicit, recorded disposition is made.
