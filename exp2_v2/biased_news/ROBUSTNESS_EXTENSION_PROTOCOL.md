# Coin City generator, rationale, and repeat-run robustness extension

Protocol version: `coin_city_robustness_v1`

Frozen on 2026-08-26 after the original Coin City behavioral and mechanistic
results were opened. All results from this extension are post-hoc robustness
checks and cannot change the original confirmatory estimands.

## R1: second generator population

The original 250 episodes use seeds 930000--930249, strong/weak slope means
0.90/0.25, and between-city slope SD 0.03. The robustness population uses fresh
seeds 940000--940249, strong/weak slope means 0.75/0.40, and slope SD 0.10.
Case-noise SD remains 5.0; episode balance, numeric ranges, prompt text, five
City-C prefixes, matched-pair construction, and episode-randomized KIV/ZOR
mapping remain unchanged. Drawn slopes are not truncated, reordered, or
conditioned on separation. Any realized regime overlap is retained and reported.

The frozen open-weight roster is Qwen3 4B/8B/14B/32B; Qwen2.5
7B/14B/32B/72B; and Llama 3.1 8B/70B. Every checkpoint receives the three
matched arms (`abc_no_context`, `abc_context`, `abc_symbol_context`) under its
original temperature-zero vLLM configuration. The primary robustness contrast
is symbol minus no-context latent-slope correlation at k=0. Forecast-MAE and
regime-accuracy contrasts are secondary. Intervals resample episodes and all ten
models are reported regardless of direction.

## R2: rationale audit

The audit uses only preserved response text and makes no model calls. It reports
rationale presence for all released response records. Content coding is limited
to k=0, where the cue is the only City-C regime information.

For the arbitrary-symbol arm, deterministic text rules identify: (a) explicit
reference-selection language; (b) a unique City A/B reference named by the
rationale; and (c) whether that reference shares City C's episode-local symbol.
For the inverted semantic arm, the same rules identify a unique A/B reference
and national/local regime words, then compare each with both the false displayed
cue and the latent true regime. Ambiguous or absent statements remain uncoded;
they are never imputed. Results are model-stratified and paired with each model's
forecast-based mapping-recovery statistic. Rationales are treated as behavioral
self-reports, not privileged evidence of internal computation.

Before writing the text rules, 50 coder-development episodes are selected as the
50 smallest SHA-256 values of `rationale-coder-v1:<episode>`. Rule development
may inspect only those episodes. The other 200 episodes form the locked audit
set and supply the primary content results; full-sample values are secondary.

## R3: GPT-5.6 run-to-run repeat

The repeat uses deployment `gpt-5.6-sol` on the `liv` Azure project, OpenAI
Responses API, low reasoning effort, provider-native sampling (no temperature or
top-p), and a 4096-token output ceiling. It replays exactly the 250 k=0 prompts
from each of the no-context, semantic-context, and arbitrary-symbol arms (750
calls). Each task receives one application-level attempt; HTTP client retries
are zero. Outputs go to a new repeat directory and never replace original data.

The analysis reports paired repeat-minus-original changes in forecast MAE,
latent-slope correlation, regime accuracy, per-task absolute prediction
difference, and parse status. Intervals resample the 250 episodes. This estimates
deployment run-to-run variability for the named snapshot only.
