# Overnight diagnostic campaign

Requested 2026-09-21 evening; work began 2026-09-22 UTC. This directory keeps
new follow-up protocols, data and outputs separate from existing experiments and
manuscript edits. Existing manuscript corrections remain untouched.

- `structure/PROTOCOL.md`: four-interface reference-selection diagnostic;
  `structure/REPORT.md` and `summary.csv` are generated as model results arrive.
  Fixed 96 paired episodes, two domains, semantic/arbitrary labels, k=0.
  Untrained Qwen3-8B, six existing RL adapters, stronger untrained Qwen3-32B.
- `evidence_use/PROTOCOL.md`: coherent low/high target-gain interventions,
  same noise and calibration systems; all twelve mechanism identities.
  A separate fresh ordinary held-out set is shared with the shuffled controls.
- `shuffled_control/`: exact target-vector multiset control, ten predeclared
  Qwen3-4B training runs (two disclosures × seeds 42–46), training-code audit,
  donor maps, source freezes and submission receipts.
- `learnability/PROTOCOL.md`: conditional small supervised pilot, frozen before
  component outputs. Same 4,800 original semantic Coin City prompts; same LoRA;
  learning rates 1e-6 and 2e-5 reported separately, seed42. Training launches only
  after `gate.json` documents that the diagnostic criterion was met.

## Important interpretation limits

The old mechanism comparator is a fixed population prior, not shuffled targets.
Checkpoint directories alone did not imply available weights: old mechanism
seeds42–44 retain scores and tokenizer files but lack final adapters. Seeds45–46
have actual weights. A fresh five-seed comparison therefore needs six matched
recovery trainings in addition to ten shuffled controls if no archived weights
are recovered. The A5000 hardware amendment additionally fixes four matched
seed45/46 reruns, making20 new training runs total (10control,10matched), so the
primary comparison changes target assignment only. Retained historical A6000
weights are a supplementary check. Historical scores cannot stand in for
fresh-set predictions.

The supervised pilot matches prompt presentations, not compute:300 SFT updates
versus approximately2400 RL updates and eight sampled completions per prompt.
One training seed cannot establish a general SFT advantage. All fixed conditions
are retained, and higher-rate-only success implicates optimization settings.

A naive predicted horizon3/horizon1 ratio can be unstable near a zero denominator.
The new structure diagnostic instead uses paired response-change projection and
response error; original registered endpoints are not silently redefined.

## Scheduler and persistence

The cluster rewrites long `all` submissions into account-owned partitions.
Short inference can run on A6000s in `all`; long A6000 training initially queued
behind unavailable/draining nodes. Available A5000s are used for inference and
are being tested with unchanged numerical training settings for the controls.
Every canceled job in the receipts belongs to this new campaign and was pending;
existing campaigns are not canceled. Source/data freezes and explicit amendments
record changes. Run status is not a result; inspect outputs and completeness.
