# Experiment 4 post-hoc architecture and provider sweep

Frozen on **2026-08-25 UTC** after the registered Qwen3-4B/8B/14B scale
results and the same-holdout Llama-3.1-8B comparator had been opened. This is a
post-hoc descriptive extension. It cannot change the original confirmatory
claims or decision rules, and every registered result will be reported.

## Questions and reporting hierarchy

1. **Smaller Qwen:** extend the solid Qwen scale trajectory with
   `Qwen/Qwen3-1.7B` at revision
   `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
2. **Second Llama:** add an unconnected Llama-3.2-3B architecture point using
   the public full-BF16 mirror `unsloth/Llama-3.2-3B-Instruct` at revision
   `006f5dcd1393c3add266de40994ba96225e9689d`. Meta's gated repository was
   inaccessible to the authenticated workspace account; the mirror identity,
   revision, and BF16 parameter inventory are recorded rather than concealed.
3. **Hosted reference panel:** evaluate `claude-opus-4-8`, `gpt-5.6-sol`,
   `FW-Kimi-K3`, and `DeepSeek-V4-Pro` as off-the-shelf systems. These points
   are never connected to the trained local-model curves and are not described
   as DAPO or base-versus-trained comparisons.

The main scaling line remains a Qwen-family result. Llama points use square
markers and remain unconnected. Hosted systems appear in a separate panel or
appendix figure labeled *off-the-shelf API references*.

## Frozen local-model procedure

Both new local checkpoints use exactly the original 1,736-task training split,
512-task development split, reward `1 - (p - y)^2`, rank-32 LoRA with alpha 64,
Dr. GRPO, learning rate `1e-6`, 4,800 prompt presentations, 300 updates, rollout
temperature 1.3, batch size 16, eight rollouts per prompt, and seeds 42, 43, and
44. Hardware colocation on one A6000 changes resource placement only; the
scientific batch, rollout, update, and optimization settings are unchanged.

A 160-presentation seed-42 canary for each checkpoint must complete and produce
all 512 development tasks with five parseable draws before its three full jobs
are eligible to run. Development and final evaluation both use five stochastic
draws at temperature .7, top-p .8, top-k 20, maximum 128 tokens, and the same
strict JSON probability contract. No deterministic/greedy endpoint is used.

The locked endpoint is the existing 318-market family-disjoint holdout with task
SHA-256
`ee9bd976402a746195291b4b43e209a084a0f1c6e5bf3f27c9d4ef638f1c7f24`.
It is absent from training and development. Incomplete five-draw task groups
receive Brier loss one. Report base, each trained seed, the three-seed mean,
trained-minus-base, and trained-minus-contemporaneous-market contrasts with the
existing connected-family bootstrap. No new local result may alter model,
checkpoint, seed, hyperparameters, or endpoint inclusion.

## Frozen hosted-model procedure

The hosted panel receives the exact same 318 user-message strings and no tools,
search, outcome, post-cutoff information, demonstrations, or training. Each
model receives five independent calls per task. The requested common controls
are temperature .7, top-p .8, and maximum 128 output tokens. If a pinned provider
deployment rejects a sampling control (notably reasoning deployments that do
not expose temperature), the runner records the provider-native setting and the
point remains in the hosted panel; it is not relabeled as decoder-identical.
Top-k is unavailable across these APIs and is therefore not requested.

Responses must contain `{"yes_prob": number}` with a value in [0, 1]. Parsing,
five-draw aggregation, incomplete-group Brier-one penalties, market baseline,
and connected-family bootstrap are identical to the local evaluator. Provider
name, endpoint family, requested/effective controls, reasoning effort, call
timestamps, response IDs when exposed, parse coverage, prompt hashes, code hash,
and raw-artifact hash are retained. One development-only transport canary per
provider may be used to establish accepted parameters; no holdout result may be
used to tune prompts or settings.

Hosted results are descriptive snapshots of the named deployments, not stable
claims about product brands such as ChatGPT or Claude generally. Because these
models were not trained by this experiment, their scores answer only how the
available off-the-shelf systems perform on the same sealed prompts.
