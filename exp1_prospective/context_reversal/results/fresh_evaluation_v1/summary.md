# Fresh Exp1 evaluation

80 finite-rule instances; eight shared mechanism classes. Model screening plus formal author adjudication, not human validation.

| Deployment | Wording | Paired reversal | Sign accuracy | Broken mean absolute pp | Valid/planned |
|---|---|---:|---:|---:|---:|
| Qwen3 enabled | base | 77/80 | 157/160 | 0.16 | 3803/3840 |
| Qwen3 enabled | names | 79/80 | 159/160 | 0.00 | 3803/3840 |
| Qwen3 enabled | paraphrase | 74/80 | 154/160 | 0.02 | 3803/3840 |
| Qwen3 enabled | resample | 75/80 | 155/160 | 0.07 | 3803/3840 |
| Qwen3 disabled | base | 14/80 | 84/160 | 4.18 | 3840/3840 |
| Qwen3 disabled | names | 14/80 | 86/160 | 2.66 | 3840/3840 |
| Qwen3 disabled | paraphrase | 12/80 | 81/160 | 2.15 | 3840/3840 |
| Qwen3 disabled | resample | 11/80 | 83/160 | 3.30 | 3840/3840 |

Base wording, thinking enabled minus disabled: 78.75 percentage points; descriptive 95% interval [68.75, 88.75].

Frontier collection: complete. Detailed control drift, failures, normative-effect bins and mechanism sensitivity are in the JSON artifacts.

| Frontier wording | Paired reversal | Sign accuracy | Broken mean absolute pp |
|---|---:|---:|---:|
| base | 24/24 | 48/48 | 0.000 |
| paraphrase | 24/24 | 48/48 | 0.000 |

Frontier usage-based cost upper bound: $1.773475; 576 planned records.

All figures use the frozen denominators. Missing, zero and invalid signed updates count as failures. Broken-link stability is indeterminate if any planned measurement is missing. The eight shared mechanisms and common outcome wrapper limit generalization to natural news; the original lexical relevance shortcut remains a limitation.

| Local mode | Recorded status | Records |
|---|---|---:|
| enabled | ok | 3803 |
| enabled | parse_error | 10 |
| enabled | blocked_baseline | 27 |
| disabled | ok | 3840 |
