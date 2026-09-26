# Experiment 3 mechanism-family large-model scale extension

Protocol version: `c3_mechanism_scale_v2_1`

Frozen on **2026-08-25 UTC** after the original Qwen3-4B, Qwen3-8B, and
Llama-3.1-8B behavioral results and the seed-42 Qwen3-8B latent-probe pilot
were known, but before any checkpoint in this extension was trained on the
mechanism-family data.  Every result is reported regardless of direction.
The already launched `mechanistic_probe/V2_PROTOCOL.md` roster is unchanged;
this document defines a prospective scale extension.

## Question and exact roster

Does the absence of a reliably decodable episode-specific linear mechanism
representation at the final input token persist at larger dense checkpoints?
The exact added checkpoints are:

- `Qwen/Qwen3-14B` at commit
  `40c069824f4251a91eefaf281ebe4c544efd3e18`;
- `Qwen/Qwen3-32B` at commit
  `9216db5781bf21249d130ec9da846c4624c16137`; and
- `meta-llama/Llama-3.1-70B-Instruct` at commit
  `1605565b47bb9346c5515c34102e054115b4f98b`.

For each model the scientific hidden-state roster is one untrained base plus
both training disclosures by both confirmatory reward arms by seeds 42, 43,
and 44: 13 endpoints per model and 39 added endpoints total.  Each endpoint is
evaluated on paired disclosed and undisclosed probe prompts.  Execution uses
36 full training jobs and six disclosure-specific base-evaluation jobs.

The separately registered Qwen3-14B Coin City structural extension and
Qwen3-32B Polymarket extension use different training data or estimands.  Their
adapters are therefore ineligible for this mechanism-family scale comparison.

## Frozen scientific settings

The extension reuses the exact original mechanism-family datasets, held-out
episodes, prompt bytes, response rewards, task manifests, and analysis split.
All scientific hyperparameters remain fixed:

- rank-32 LoRA with alpha 64; Dr. GRPO; beta 0;
- learning rate `1e-6`, constant schedule, and 3% warmup;
- 4,800 presentations, 16 prompts per round, eight rollouts per prompt, and
  300 rounds;
- rollout temperature 1.3 and top-p 1;
- maximum model length 3,072, prompt limit 2,304, and generation limit 192;
- bf16, gradient checkpointing, FlashAttention, reference offload, and the
  registered syntax-only XGrammar forecast-array contract; and
- the original greedy monitoring evaluation plus five endpoint draws at
  temperature 0.7, with draws averaged within task.

The two reward arms have byte-identical prompts.  Only the reward target
differs.  The training seed remains the top-level replication unit.

## Memory-only allocation profiles

Only memory distribution changes with checkpoint size:

| Model | GPUs | Rollout actor | Learner | ZeRO | vLLM ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3-14B | 2 A6000 | TP1 | 1 rank | 2 | .78 |
| Qwen3-32B | 4 A6000 | TP2 | 2 ranks | 3 | .78 |
| Llama-3.1-70B | 8 A6000 | TP4 | 4 ranks | 3 | .78 |

ZeRO-3 profiles disable the fused LM head and use LoRA-only synchronization.
Per-learner rollout batches are 16, 8, and 4 respectively, preserving the
global batch of 16 prompts and 128 rollout transitions.  These are execution
settings, not scientific hyperparameters.

## Feasibility gate

Before full training, each model runs seed-42 causal-family canaries for both
training disclosures: six canaries total.  A canary uses exactly 160
presentations (ten rounds), evaluates the complete registered held-out grid,
and is excluded from every estimate.  Six untrained base evaluations run in
parallel and are scientific endpoints, not canary estimates.

The full 36-job roster is released automatically only if all canaries save a
valid adapter, have finite nonconstant reward traces, keep training truncation
at or below 5%, and achieve at least 95% syntax parse coverage.  The gate has
no accuracy, MAE, mechanism-classification, gain, or probe threshold and may
not be tuned from canary performance.  A failed resource profile requires a
written allocation-only amendment before replacement.

## Latent analysis

After every full endpoint completes, the frozen v2 analysis is applied without
model-specific tuning:

- development-only world residualization and the development-estimated
  world-mean primary baseline;
- separate and pooled evidence-depth probes;
- displayed-number and oracle-posterior positive controls;
- direct and development-mean-aligned cross-disclosure transfer; and
- sealed per-episode predictions for seed-first and paired-episode inference.

Models are fitted and summarized separately; hidden coordinates are never
pooled across architectures.  The primary large-model inference is consistency
of the causal-family-minus-population-prior contrast across the three training
seeds within each checkpoint.  Scale contrasts compare 8B versus 14B versus
32B within Qwen and 8B versus 70B within Llama.

## Claim boundary

The extension can support a claim across the evaluated dense Qwen and Llama
sizes.  It cannot establish absence of nonlinear information, absence at other
token positions, or universality over untested architectures.  If numerical
positive controls fail at a checkpoint, its result is reported as an
extraction-location limitation rather than a selective mechanism null.
