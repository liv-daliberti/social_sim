# Experiment 4 Qwen scale extension

Protocol version: `exp4_qwen_scale_v1`

This protocol was frozen after the original Qwen3-4B, Qwen3-8B, and
Llama-3.1-8B locked tests were opened and before the Qwen3-14B holdout was
built, any Qwen3-14B training began, or any model was evaluated on the new
holdout. The extension is a new-holdout scale replication, not a continuation
of the original sealed test.

## Question and roster

Does the base-relative forecasting gain replicate at Qwen3-14B, and does the
trained Qwen3 series improve with scale from 4B to 8B to 14B?

The fixed roster is Qwen3-4B-Instruct-2507, Qwen3-8B, and Qwen3-14B. Existing
4B and 8B round-300 adapters are reused without further training. Qwen3-14B is
trained at seeds 42, 43, and 44. Every model uses its non-thinking chat
template. Because the 4B checkpoint is the later Instruct-2507 revision, the
controlled same-release scale contrast is 8B versus 14B; the 4B point is
descriptive support rather than evidence for a pure three-point parameter
scaling law.

## New holdout

The builder rescans the same frozen 1,788,703-row archive used by the registered
Experiment 4 protocol. It begins with the same eligibility, censoring, domain,
30-day forecast horizon, and chronological rules. Candidate selection never
uses the settlement label or market forecast.

The new holdout contains every eligible 2026 test-window task whose event ID,
available series ID, normalized question template, and task ID are absent from:

- every eligible pre-test train/development family used to construct the
  registered split; and
- every task and family in the already-opened 1,024-market test.

Rows are ordered by a SHA-256 hash of the protocol version and task ID. The
builder reports counts, dates, family counts, and hashes but no labels, outcome
rates, market scores, or model scores. The holdout path is never supplied to a
training process. A minimum of 256 tasks is required.

## Qwen3-14B training

Optimization, rollout, and training-budget hyperparameters remain fixed to the
completed 4B/8B protocol:

- rank-32 LoRA, alpha 64; Dr. GRPO; beta 0;
- learning rate `1e-6`, constant schedule, 3% warmup;
- the same deterministic 4,800-presentation training schedule;
- 16 prompts per round, eight rollouts per prompt, 300 rounds;
- rollout temperature 1.3 and top-p 1;
- pure reward `1 - (p - y)^2`;
- bf16, gradient checkpointing, FlashAttention, ZeRO stage 2, and reference
  offload; and
- the same 512-task development split evaluated every 25 rounds with five
  stochastic non-thinking draws per task.

Only execution settings change for memory: one A6000 rollout worker and one
A6000 learner, vLLM memory utilization 0.78, 100 GB host memory, and a 30-hour
limit. A ten-round seed-42 canary must complete, save a valid adapter, retain
finite metrics, and parse all development outputs before the three full jobs
may be submitted. Canary output is never used as a paper result.

## Locked stochastic evaluation

The new holdout is opened only after all three Qwen3-14B seeds complete. Three
dependency-locked jobs evaluate 4B, 8B, and 14B separately. For every base and
trained endpoint, evaluation uses five non-thinking draws per task with
temperature 0.7, top-p 0.8, top-k 20, a fixed sampling seed, and at most 128
new tokens. Probabilities are averaged within a task only when all five draws
parse. If any draw is invalid, that endpoint/task receives the registered
missing-output Brier loss one. Raw draws and both draw-level and complete-task
parse coverage are retained.

## Estimands and reporting

The primary contrast is the three-seed mean Qwen3-14B trained Brier loss minus
its untrained base on the new holdout. Secondary contrasts are:

- trained Qwen3-14B minus trained Qwen3-8B;
- trained Qwen3-14B minus the latest genuine pre-cutoff market price;
- trained Qwen3-14B minus the train-only Platt calibration frozen in the
  original protocol; and
- the 4B/8B/14B trained and base scale profiles.

Intervals use the registered 5,000-repetition connected-family bootstrap.
Negative deltas favor the first named model. A market-superiority claim
requires the upper confidence bound for trained-minus-market Brier below zero;
Platt superiority is reported separately under the same rule. The complete 14B
result and scale profile are reported regardless of direction.

## Fail-closed execution

The dataset manifest records the archive, registered-manifest, builder,
protocol, and output hashes. The launcher verifies all hashes, the fixed model
roster, cached checkpoints, existing adapter identities, seed set, canary state,
and dependency graph. Evaluation summaries record holdout and adapter hashes.
Any missing model, seed, task, draw, adapter, family constraint, or registered
baseline aborts the handoff.
