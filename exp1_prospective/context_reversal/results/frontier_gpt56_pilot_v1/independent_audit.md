# Independent GPT-5.6 frontier artifact audit

Passed 2902 checks: 320 unique planned outcomes, all valid, no missing records.

Returned model: {'gpt-5.6-sol': 320}. Estimated cost: $0.712128.

Tokens: {'cached_input_tokens': 0, 'input_tokens': 101072, 'output_tokens': 15392, 'reasoning_tokens': 9976, 'total_tokens': 116464}.

| Outcome | Raw | Drift adjusted |
|---|---:|---:|
| Positive direction | 19/20 | 19/20 |
| Negative direction | 18/20 | 18/20 |
| Pooled direction | 37/40 | 37/40 |
| Paired reversal | 17/20 | 17/20 |

Exact frozen prompts, own-context baseline priors, all provenance hashes, unique response IDs, request ledger, per-row usage, per-row cost, and manifest totals agree.

Broken raw movement: mean -0.500 pp; mean absolute 0.500 pp; equivalence criterion True.
Broken drift-adjusted movement: mean -0.500 pp; mean absolute 0.500 pp; equivalence criterion True.

| Context | No-news mean / absolute pp | Repeated-news mean / absolute pp |
|---|---:|---:|
| positive | 0.000 / 0.000 | 0.000 / 0.000 |
| negative | 0.000 / 0.000 | 0.000 / 0.000 |
| broken | 0.000 / 0.000 | 0.000 / 0.000 |
| masked | 0.000 / 0.000 | 0.000 / 0.000 |

All intervals use 2,000 whole-family resamples with seed 20260921. Outcomes remain exploratory development diagnostics.

Retained direction failures: dev_07 negative 0.62 to 0.68 (+6 pp); dev_10 positive 0.50 to 0.35 (-15 pp); dev_18 negative 0.45 to 0.58 (+13 pp). No directional baseline lies at an endpoint that prevents the scored direction. The sole nonzero broken-link update is dev_16, 0.30 to 0.20 (-10 pp).

The destination of outward material flow is unspecified in dev_07, dev_10, and dev_18. Alternative destination readings can change the inferred sign; dev_10 shares this ambiguity with dev_06. These are post hoc material caveats. All labels, cases, and denominators remain unchanged.
