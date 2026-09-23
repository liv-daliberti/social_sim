# Triple-level analysis of the frozen context-reversal cohort

Generated 2026-09-23T14:28:24.522432+00:00 from `probability_plan.jsonl` (sha256 `0be8994b7b6edfc4`). Offline; no model was called.

## Design

320 triples over 960 units. Within a triple the evidence text, the prior and the named entity are identical and only the stated relation differs; all five invariants were verified: identical evidence text, identical prior, identical entity, relations cover {-1, 0, +1}, three distinct correct answers.

## The bound

A predictor that reads only the evidence text, the prior and the named entity is constant within a triple. The three correct posteriors are distinct, so it matches at most one of three items and never a whole triple. Maximum item accuracy 1/3, maximum triple accuracy 0.

| Predictor | Items, direction | Triples, all three |
|---|---:|---:|
| Predict no change | 320/960 | 0/320 |
| Follow the evidence direction | 320/960 | 0/320 |
| Always revise upward | 320/960 | 0/320 |
| Best constant, given every oracle | 320/960 | 0/320 |

## Systems

| System | Triples (uncond.) | Triples (complete) | Items direction | Items exact | Posterior err. mean (pp) | Inert exactly zero |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 48/48 | 48/48 | 144/144 | 144/144 | 0.000 | 48/48 |
| Claude Opus 5 | 17/48 | 17/17 | 78/144 | 54/144 | 0.013 | 29/29 |
| Qwen3-32B thinking off | 29/320 | 29/320 | 519/960 | 1/960 | 28.220 | 185/320 |
| Qwen3-32B thinking on | 297/320 | 297/310 | 937/960 | 811/960 | 0.129 | 312/316 |

## Unconditional accounting

Every planned call appears in the denominator. Model failures, instrument failures and failures cascading from a failed baseline are reported separately and none is dropped.

| System | Planned | Valid | Model failure | Instrument failure | Cascade | Pending |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 576 | 576 | 0 | 0 | 0 | 0 |
| Claude Opus 5 | 576 | 339 | 84 | 0 | 153 | 0 |
| Qwen3-32B thinking off | 3840 | 3840 | 0 | 0 | 0 | 0 |
| Qwen3-32B thinking on | 3840 | 3803 | 5 | 5 | 27 | 0 |

Unconditional denominators count refusals, truncations and blocked updates as failures. Exact agreement is to four decimals, the granularity the oracles are stated at.
