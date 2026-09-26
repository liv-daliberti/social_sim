# Structure-selection component diagnostic, v1

Frozen before diagnostic inference on 2026-09-22 UTC (2026-09-21 local evening).
This is an inference-only diagnostic; it does not replace registered endpoints.

## Fixed evaluation

96 fresh latent episodes: 24 per Coin City/Coin Harbor × semantic/arbitrary-label
cell, from seed band 171,000,000–172,100,023. Each episode has two cue versions,
selecting direct or mediated reference with identical numerical reference evidence
and identical reference order. Reference parameters, noise, arbitrary mappings,
and reference order come from the existing structure-selection generator.
All tasks have zero target observations: this isolates selection and arithmetic,
without introducing conflicting target trajectories when the cue is flipped.
Training and previous evaluation use distinct seed bands (61M and 71M/76M).
There are 192 tasks × four interfaces = 768 prompts per model.

1. **Selection only:** retain reference trajectories and backgrounds, replace the
   forecast question with identification of reference 1 or 2 from the target cue.
2. **Original:** exact upstream task prompt with paired target-cue flips.
3. **Both patterns supplied:** original prompt plus accurate per-unit impulse
   response summaries for both references at horizons 1 and 3. The model still
   selects and multiplies by the requested scenario shock.
4. **Selected reference and pattern supplied:** original prompt plus the correct
   reference number and its per-unit response summary. Selection is solved.

Summaries come from the simulator, are rounded to eight decimals, and reveal no
internal mechanism names. They give unit effects, not the ten forecast answers.
A deterministic invariant verifies summary-based arithmetic reproduces all
simulator forecasts to 1e-6. Paired original prompts differ only in target
background. These summaries are idealized assistance, not claimed estimates
extractable without uncertainty from fourteen noisy observations.

## Fixed models and decoding

Untrained Qwen3-8B; its six existing step_00301 LoRA checkpoints, episode-matched
(`causal`) and fixed population prior, seeds 42/43/44; and untrained Qwen3-32B as
stronger feasibility comparison. Adapter paths and weight hashes are in roster.
Run base 8B, matched s42, then the remaining checkpoints, writing results after
each. Existing training jobs and registered endpoints are unchanged.

Use the same vLLM runtime and chat template as existing endpoints, thinking off,
one greedy draw, seed 20260921. Constrain only JSON syntax: reference integer
1/2 for selection, exactly ten finite forecast numbers for other interfaces.
Maximum output 32 tokens for selection and 192 for forecasts; context 4096.
No multiple-choice mechanism labels or gold forecasts constrain decoding.

## Metrics and interpretation

Report every domain × label × interface cell. Selection accuracy and rate of
both paired cues correct test cue-to-reference identification. Forecast response
MAE (relative to zero-shock prediction at each horizon) and ordinary forecast
MAE assess numerical use; nearest-template structure accuracy is secondary.
For each paired cue flip, compare the change in predicted response vector with
the simulator's change. Calibration is their projection coefficient: 0 means
cue invariance, 1 means the expected change. Also report change-vector MAE and
positive-direction rate. Report parse rate separately; invalid forecasts receive
the domain range penalty in MAE and zero accuracy. Cue-change metrics require
both parses and disclose their count. CIs resample paired episodes within each
cell (2,000 bootstrap draws, fixed seed). These are diagnostic episode CIs,
not evidence for a seed-generalized trained-arm contrast.

Selection failure locates a basic interface/label bottleneck. Selection success
with original failure but both-pattern success implicates extracting response
patterns. Failure even when the selected pattern is supplied implicates numerical
use or instruction/output adaptation. Stronger-model success establishes LM
interface feasibility at this scale. Domain and arbitrary-label breakdowns
locate transfer boundaries. k>0 target-evidence integration remains untested.
Any later SFT/RL comparison is separately frozen and reported as diagnostic.

## Integrity and resource budget

`build_diagnostic.py` freezes data, protocol, code and adapter hashes before jobs.
`--validate-only` rechecks construction invariants without changing artifacts.
One single-A6000 job sequentially evaluates the 8B base and six LoRAs; one
2×A6000 job evaluates 32B with tensor parallelism. Each has an 8h maximum.
No training runs are part of this protocol.
