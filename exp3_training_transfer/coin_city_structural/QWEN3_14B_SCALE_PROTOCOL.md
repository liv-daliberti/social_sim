# Experiment 3 Qwen3-14B scale extension

Protocol version: `coin_city_qwen3_14b_scale_v1`

Frozen on **2026-08-25 UTC** after the original Experiment 3 Qwen3-4B,
Qwen3-8B, and Llama-3.1-8B results were known, but before Qwen3-14B was
trained or evaluated on the registered 1,440-task Coin City transfer endpoint.
This is a prospective scale extension, not part of the original confirmatory
roster. Its complete result is reported regardless of direction.

## Question and fixed roster

Does the population-prior-minus-episode-matched transfer advantage replicate
at Qwen3-14B? Secondarily, is that advantage different between the same-release
Qwen3-8B and Qwen3-14B checkpoints?

The extension adds exactly `Qwen/Qwen3-14B`, with thinking disabled through its
tokenizer chat template. The confirmatory roster is two reward arms
(`causal`, the legacy artifact name for episode-matched reward, and
`population_prior`) by seeds 42, 43, and 44, plus one untrained base endpoint.
There is no Qwen3-14B structureless arm. Qwen3-4B is descriptive context rather
than a controlled scale point because it is the later Instruct-2507 revision;
Qwen3-8B versus Qwen3-14B is the controlled same-release scale comparison.

## Frozen data and training

The extension reuses the exact original Experiment 3 data. Each trained run
uses the same 4,800-presentation schedule, and every endpoint uses the same
1,440 held-out tasks. The launcher verifies the original passed preflight
manifest, each canonical saved-dataset shard and build manifest recorded there,
the canonical original result and its ledger, and the cached Qwen3-14B
checkpoint before submission. Regenerable Hugging Face `cache-*.arrow` files
are explicitly excluded; saved-dataset state points only to the pinned
`data-*.arrow` shards.

All scientific hyperparameters remain fixed to the original Qwen3-8B runs:

- rank-32 LoRA, alpha 64; Dr. GRPO; beta 0;
- learning rate `1e-6`, constant schedule, 3% warmup;
- 16 prompts per round, eight rollouts per prompt, 300 rounds;
- rollout temperature 1.3 and top-p 1;
- the registered response reward and the same prompt/reward data;
- bf16, gradient checkpointing, FlashAttention, ZeRO stage 2, and reference
  offload; and
- maximum model length 3,072, prompt limit 2,304, and generation limit 192.

Only the execution allocation changes. The excluded canary and untrained base
endpoint ran on A100s. Before any confirmatory training began, the six-run
allocation was administratively amended to one A6000 rollout worker and one
A6000 learner under the `allcs` account and `cs` partition. No data,
hyperparameter, endpoint, or estimand changed. vLLM memory utilization remains
0.78 with expandable CUDA segments. One ten-round, seed-42 `causal` canary
must
finish, save a valid adapter, retain
finite nonconstant rewards, and complete both held-out decodes before the six
scientific training jobs are submitted. Canary output is excluded from every
estimate.

## Endpoint and estimands

Every trained checkpoint and the untrained base are evaluated on the frozen
held-out tasks with:

- the registered temperature-zero decode, retained for provenance; and
- the manuscript endpoint of five non-thinking draws at temperature 0.7, with
  the original fixed seeds and grammar-constrained forecast-array contract.

Malformed outputs receive the original registered full-range error penalty.
Draws are averaged within task. The training seed is the top-level replication
unit, with paired task resampling within seed and 5,000 bootstrap repetitions.

The primary estimand is population-prior minus episode-matched response MAE in
the joint-shift cell: Coin Harbor, unseen mediated Structure B, correct pathway
hint, and equal weight over evidence depths `k in {0,2,4,8}`. Positive values
favor episode-matched reward.

Prespecified secondary outputs are:

- the same contrast in the other three domain-by-structure cells;
- Qwen3-14B cue-absence and misleading-cue penalties, overall and by `k`;
- the `k=8` minus `k=0` change in misleading-cue penalty; and
- the Qwen3-14B minus Qwen3-8B difference in the primary training contrast,
  paired by training seed and task. Positive values mean a larger
  episode-matched advantage at 14B.

The renderer validates the exact 7-job/14-file extension roster, 60,480 scored
row-draws, endpoint identities, the complete shared task universe, source and
data hashes, and the frozen Qwen3-8B parent score files before writing a JSON
result or TeX table.

## Interpretation

The original Experiment 3 primary claim remains Qwen3-8B. The 14B result is
identified as a post-original, prospectively frozen scale extension. Its
direction does not determine whether it is included: a positive, null, or
negative result is reported under the same analysis.
