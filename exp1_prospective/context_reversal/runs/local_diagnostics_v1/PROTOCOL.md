# Local diagnostic follow-up, version 1

This is an exploratory follow-up authorized after inspecting the completed local
and frontier pilots. No frontier API calls or new human-review round are part of
this run. Existing families, annotations, and prior results remain unchanged.
The user reports that material review is already complete. Review provenance is
recorded separately; no new claim about review coverage is inferred from this run.

## Shared design

All 20 original families and all four contexts, repeat zero only. The probability
plan is byte-identical to the frontier pilot plan. Each task uses the same visible
scenario and messages, with no private labels, model outputs, or explanations
from another arm. Context labels never enter the user prompt. Nothing is excluded
based on prior performance. Masked contexts have no assigned sign or null target.

## Direction companion

Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct, and Qwen3-32B (non-thinking) each
classify 240 independent scenario-message combinations: 80 context units times
new news, no news, and repeated information. No numeric baseline is elicited or
shown. Choices are increase, decrease, unchanged, and unclear. The new-news
primary diagnostic denominators are 40 directional trials, 20 both-correct pairs,
and 20 broken-link trials. No-news/repeated-news controls each include 80 trials.
Unclear and invalid/missing outputs remain failures against determinate targets;
report their rates separately. Masked new-news outputs are descriptive only.
Use temperature 0.7, top-p 1, 128 output tokens, constrained categorical JSON, and
the same local checkpoint snapshots as the original pilot. The direction task is
a changed elicitation task, not a replacement for forecasting accuracy.

Join predictions to each model's saved repeat-zero probability responses. Report
joint direction/numeric success and endpoint blocking. Any conditional analysis
is secondary and keeps the original unconditional scores intact. Family bootstrap
intervals use the existing 2,000-draw seed and retain all contexts/arms.

## Qwen3 reasoning comparison

Run the same Qwen3-32B checkpoint in two arms: thinking enabled and thinking
disabled. Both use unconstrained generation, strict parsing of final probability
JSON, temperature 0.7, top-p 1, 4,096 output tokens, maximum sequence length 8,192,
and identical request seeds and batch size. Enabled prompts use the model's thinking-enabled assistant prefix (empty in the
cached tokenizer, so the model emits the opening tag); disabled prompts use its
explicit empty thinking block. The runner verifies these prefixes differ.
Record token counts, reasoning presence and closure, raw output, parse failures,
and length truncation. A valid final probability remains valid if the model
chooses an empty thinking section. Do not extract a probability from unfinished
reasoning. Both arms independently elicit their own 80 context-specific baselines
and 240 forked updates. Each update sees only its own arm's context baseline.

The extra unconstrained non-thinking arm controls for removing the original
probability-only grammar and increasing the token allowance. The original
constrained non-thinking run remains a separate reference. This isolates a
thinking-mode deployment change more closely; it does not equate local and
frontier inference or expose the model's internal mechanism.

Score with the original probability analyzer and 2-point broken-link stability
criterion, retain all failures, and report all 320 planned records per arm.
Infrastructure recovery may resume missing calls under unchanged configuration;
it may not replace a valid response because of its value. Freeze code and input
hashes before inference. Outputs and manifests are stored in distinct paths.
