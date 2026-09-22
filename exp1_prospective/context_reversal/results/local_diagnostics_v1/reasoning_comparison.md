# Qwen3 reasoning-mode comparison

Status: **complete_with_invalid_responses**.

Exploratory development diagnostic. New-news minus each mode’s own context baseline defines all primary outcomes. Binary failures retain planned denominators; missing magnitudes are never replaced by zero.

Enabled minus disabled contrasts use 2,000 paired whole-family bootstrap draws, seed 20260921. Values and 95% intervals are in percentage/probability points.

| Metric | Disabled | Enabled | Enabled minus disabled | Paired families |
|---|---:|---:|---:|---:|
| sign_accuracy | 62.50 [50.00, 75.00] | 75.00 [60.00, 87.50] | 12.50 [-2.50, 27.50] | 20 / 20 |
| paired_reversal | 30.00 [10.00, 50.00] | 60.00 [35.00, 80.00] | 30.00 [5.00, 55.00] | 20 / 20 |
| broken_mean_absolute_movement_pp | 14.05 [7.55, 21.30] | 7.63 [2.65, 13.16] | -7.11 [-16.32, 1.32] | 19 / 20 |

Broken common-pair means: disabled 14.74 [8.05, 22.22]; enabled 7.63 [2.65, 13.16]. Marginal missing families: disabled 0, enabled 1; missing pairs 1.

| Mode | Present / planned | Status counts | Reasoning present / received | Empty reasoning | Truncated | Output tokens | Mean output tokens | Direction-blocked endpoints / valid baselines |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| disabled | 320 / 320 | ok=320 | 0 / 320 | 0 | 0 | 3994 | 12.48125 | 5 / 40 |
| enabled | 320 / 320 | blocked_baseline=15, ok=300, parse_error=5 | 305 / 305 | 0 | 5 | 294758 | 966.4196721311475 | 1 / 37 |

disabled failure kinds: {}; missing directional baselines: 0; reasoning closed: 0; mean reasoning characters: 0.0.

enabled failure kinds: {"blocked_baseline": 15, "truncated": 5}; missing directional baselines: 3; reasoning closed: 300; mean reasoning characters: 4280.350819672131.

The original constrained nonthinking Qwen3 repeat-0 result is a separate descriptive reference (128 output tokens and probability grammar):

Sign accuracy 62.50 [50.00, 75.00]%; paired reversal 30.00 [10.00, 50.00]%; broken mean absolute movement 13.25 [8.00, 18.75] probability points. It is excluded from the paired contrast.

Reference: `/n/fs/similarity/social_sim/exp1_prospective/context_reversal/results/frontier_gpt56_pilot_v1/comparison.json`; SHA-256 `0518a26dacf5976acef41c9f50bdd230e09ba0b7d0ddf52fe0afc13c519cd515`.

Definition SHA-256: `3a94cec4d620673fb2f5da70dd5a6b287436ef2106c915e69abc009e9b9a81cd`. JSON retains exact response/manifest/code provenance, all paired family values, coverage, failures, endpoints, token usage, and truncation diagnostics.


## Symmetric formatting amendment

Both modes accept only one optional surrounding Markdown json or unlabeled fence around strict final JSON; enabled still requires the thinking close.
Reuse every already-received completion; generate only previously blocked unattempted updates; no new baselines or retries of malformed/truncated outputs.
All original paired-comparison definitions, denominators, and bootstrap draws/seed are unchanged.

| Original strict v1 arm | Valid / planned | Status counts | Strict sign accuracy | Strict paired reversal | Fenced outputs recovered | Newly unblocked update requests |
|---|---:|---|---:|---:|---:|---:|
| disabled | 19 / 320 | blocked_baseline=198, ok=19, parse_error=103 | 0.00 [interval unavailable]% | 0.00 [interval unavailable]% | 103 | 198 |
| enabled | 300 / 320 | blocked_baseline=15, ok=300, parse_error=5 | 75.00 [60.00, 87.50]% | 60.00 [35.00, 80.00]% | 0 | 0 |

Amendment: `/n/fs/similarity/social_sim/exp1_prospective/context_reversal/runs/local_diagnostics_v1/FORMAT_RECOVERY_AMENDMENT.md`; SHA-256 `7a29e2dde2b7f0317bc12b4af09f0edc9b834746058797fadd1a71c6ecc97831`.
