# Experiment 3B registered protocol

Protocol version: `exp3b_registered_v1`

## Question

Does proper-score training on historical forecast questions improve a fixed base
model on unseen 2026 question families? Does any gain survive comparison with both
the contemporaneous market and a train-only market-calibration baseline?

## Frozen sample

The builder fully scans the 1,788,703-row archive at SHA-256
`5697f7672bcb8dcc0af73272ff9ff1ec662241e92b0ac84f6e2bd4ba8f638e7d`.
It selects politics, government, macroeconomic, and geopolitical questions using
boundary-aware phrase matches, with hard exclusions for culture, awards, sports,
esports, and crypto. Current tags are used only for domain selection and never
enter a model prompt.

| Split | Decision time | Unique markets | Use |
|---|---|---:|---|
| train | before 2025-07-01 UTC | 1,736 | gradients and Platt fit |
| dev | 2025-07-01 through 2025-12-31 | 512 | training curves only |
| test | 2026-01-01 through 2026-06-30 | 1,024 | one locked evaluation |

Dev and test exclude any event ID, available official series ID, or normalized
question template observed earlier. The selected splits have zero overlap on these
keys. No selected row has an available official series ID; event and template
keys therefore provide the operative grouping in this archive.

## Unit and censoring

- One example per resolved Polymarket market.
- Outcomes must be literal `Yes`/`No` labels with a `price_extreme` settlement.
- The forecast cutoff is 30 days before the earliest archived close, scheduled
  end, or resolution timestamp.
- Inputs contain only question text, resolution rules, cutoff, and at most 14
  genuine pre-cutoff YES closes, with no more than one close per UTC date.
- Empty/carry-forward and post-cutoff candles, final prices and volume, category
  tags, realized close times, resolution fields, and status flags never enter the
  model input.
- A market needs at least four distinct genuine daily prices, a latest price no
  more than seven days stale, and a cutoff price in `[0.02, 0.98]`.
- The archive lacks revision history for static question/rules text. These fields
  are treated as time-invariant, and this residual limitation is reported.

The fail-closed preflight verifies file hashes, split dates, task uniqueness,
family disjointness, prompt/reference separation, one close per date, schedule
balance, and exact Qwen token lengths. The longest wrapped prompt is 1,012 tokens
against a 1,792-token limit.

## Training

- Base: `Qwen/Qwen3-4B-Instruct-2507`.
- Rank-32 LoRA, alpha 64; Dr. GRPO; learning rate `1e-6`.
- The 1,736 unique train markets form a deterministic 4,800-presentation
  schedule: every market appears twice and 1,328 appear a third time.
- Batch 16, eight rollouts per prompt, temperature 1.3, and exactly 300 updates.
- Pure reward `1 - (p - y)^2`; no format, rationale, or grounding bonus.
- Seeds 42, 43, and 44. Development evaluation occurs every 25 updates at
  temperature zero. The training script never receives the test path.

## Locked model-roster extension

The confirmatory roster adds `Qwen/Qwen3-8B` and
`meta-llama/Llama-3.1-8B-Instruct` without changing the registered data, train
schedule, reward, optimizer, LoRA configuration, batch/rollout counts, update
count, seeds, development schedule, or locked-test decision rule above. Qwen3-8B
uses its non-thinking chat template; Llama uses its native instruct template.
Each seed uses two A6000 GPUs (one rollout worker and one learner), 100 GB host
memory, eight CPUs, bf16, ZeRO stage 2, gradient checkpointing, FlashAttention,
a 1,920-token model context, 1,792 prompt tokens, and at most 128 generated
tokens. The synthetic C3 canary audit must pass before submission.

For each added model, one dependency-locked evaluation loads the untrained base
and all three final adapters and makes exactly one greedy (`temperature=0`) call
per system on each of the same 1,024 sealed test tasks. Invalid outputs retain
Brier loss 1. The evaluator recomputes the registered trained-minus-base,
trained-minus-market, and trained-minus-train-only-Platt contrasts with the same
connected-component family bootstrap. The test file is supplied only to this
post-training evaluator.

The fail-closed launch and paper-rendering path is:

```bash
.runtime/oat_conda/bin/python \
  exp3_training_transfer/polymarket/scripts/launch_model_extension.py

.runtime/oat_conda/bin/python \
  exp3_training_transfer/polymarket/scripts/render_model_extension_table.py \
  --latex paper/tables/exp3b_model_roster_results.tex \
  --latex paper/ICLR/tables/exp3b_model_roster_results.tex
```

The launcher records frozen-input and script hashes, complete Slurm commands,
environment variables, job IDs, and the successful synthetic-canary audit hash
in `runs/`. The renderer validates both extension summaries against the frozen
Qwen3-4B result, writes an aggregate JSON with input hashes, and emits
byte-identical tables into both paper trees. It refuses missing models, missing
seeds, altered test counts, altered baselines, or absent registered contrasts.

## Comparisons and decision rule

Invalid model outputs receive Brier loss 1 and are included in parse coverage.
Baselines are the untrained base, latest genuine market price, train prevalence,
and logistic recalibration of the market logit fit on train only. Confidence
intervals bootstrap connected components over event, series, and normalized
question-template keys.

The raw market's locked-test Brier is 0.11369. Train-only Platt calibration reaches
0.11232; its Brier delta relative to the raw market is -0.00137 with a family-
bootstrap 95% interval of [-0.00262, -0.00011]. This is the strict non-neural
comparator and was frozen before model training.

The primary learned contrast is three-seed mean trained loss minus base loss on the
locked test. A strong market-improvement claim additionally requires the upper 95%
confidence bound for trained minus raw-market Brier below zero. Beating Platt is
reported separately as the stricter evidence that text-conditioned learning adds
more than global market recalibration.

The initial pending submission (jobs 30459333--30459336) was cancelled before
allocation while an additional prompt-identity invariant was audited. That audit
found zero mismatches across all 3,272 examples and is now enforced by preflight.
Replacement jobs 30459655, 30459656, and 30459657 implement the registered seed
triplet. Job 30459658 is dependency-locked and evaluates the base model plus all
three adapters exactly once only if every training job succeeds. Timestamped
submission ledgers are retained in `runs/`.

A separate post-training analysis applies the fixed Experiment 1 evidence
interventions and reports EHC, HFC, and sensitivity. Every condition receives the
same canonical initial structured forecast for each of 100 markets and the same
nine packets per market, isolating conditional updating from retrieval and
initial-world-model construction. That secondary analysis is not part of the
accuracy job and cannot change the locked-test decision rule.

Array job 30535058 completed the base and all three final adapters with 900/900
structured parses per condition. Three-seed mean EHC/HFC is 0.9191 versus 0.9090
for base; the paired market-bootstrap difference is +0.0101 with 95% interval
[+0.0004, +0.0207]. Mean sensitivity is 4.834 versus 4.642; its difference of
+0.192 has interval [-0.073, +0.482]. Thus training modestly improves directional
consistency, while the sensitivity change is not distinguishable from zero.
