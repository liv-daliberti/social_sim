# Amendment: output-token budget for the enabled-thinking arm

Recorded 2026-09-22T17:14:52.038180+00:00. Machine-readable record: `TOKEN_BUDGET_AMENDMENT.json`.
Amended freeze: `data/fresh_evaluation_v1/frozen_v1_amended/local_freeze_max_tokens_15360.json`.

## What changed

`local_settings.max_tokens` 8192 -> 15360, for the **enabled arm only**.
Nothing else changed: materials, plans, prompts, scoring, per_request_seeds, temperature, top_p, max_model_len, batch_size, tensor_parallel_size, dtype, model_snapshot.

## Why

The original freeze gave both reasoning arms the same 8192-token output allowance.
That single number does opposite things in the two arms.

| Arm | Records | Output tokens | Terminated by length |
|---|---:|---|---:|
| disabled | 3840 | p50 14, max **31** | 0 |
| enabled | 848 | p50 5052, p90 7604, p99 8121 (completed only) | **163 (19.2%)** |

Every truncated enabled response sits at exactly 8192 tokens. The completed
enabled distribution does not taper toward the ceiling, so this is censoring by
the instrument, not a property of the model's behaviour.

The censoring is structured, not random:

| Mechanism class | Truncated | Rate |
|---|---:|---:|
| `time_windows` | 46/111 | 41.4% |
| `bounded_feedback` | 37/101 | 36.6% |
| `evidence_dependence` | 32/106 | 30.2% |
| `directed_routes` | 21/101 | 20.8% |
| `calibration` | 14/96 | 14.6% |
| `procedural_gates` | 7/113 | 6.2% |
| `broker_allocation` | 5/106 | 4.7% |
| `quorums` | 1/114 | 0.9% |

The design balances these eight classes deliberately. The allowance was deleting
responses hardest in the classes that demand the most reasoning, which is the
phenomenon this arm exists to measure. Scoring those records as model failures,
as the protocol requires for genuine failures, would charge an instrument limit
to the model, differentially by condition.

All 848 affected records were `condition: baseline`. A truncated baseline
blocks that family's three updates, so the loss propagates into every paired
comparison downstream.

## Why the new value, and why it costs nothing

Longest prompt observed across both arms and all four conditions:
**617 tokens**, against a frozen 16,384-token context.
15767 output tokens were available; 8192 were being used.
617 + 15360 = 15977 stays inside the context window, so
`max_model_len`, KV cache, concurrency and memory are all unchanged. The residual
truncation rate at the new allowance cannot be predicted from a censored sample
and will be reported as measured.

## Why the arms remain matched

The disabled arm is retained unchanged under the original freeze. Its longest
response was 31 tokens against a 8192-token allowance, so the allowance
is provably non-binding there: with per-request seeds fixed by
SHA256([base_seed, trial_id, stage, condition]) and every sampling parameter
unchanged, a larger allowance cannot alter a single disabled response. The two
arms therefore differ in no setting that can affect generation.

## Integrity

Only instrument fields were inspected to justify this amendment:
finish_reason, output_token_count, prompt_token_count, condition, domain. No probability, update, reversal or direction
outcome was computed from target responses beforehand. The superseded enabled
responses are preserved at `responses/fresh_evaluation_v1/_superseded_max_tokens_8192_20260922`
and are not deleted. The original freeze, its materials, plans, scoring code and
all recorded hashes are untouched and still verify.
