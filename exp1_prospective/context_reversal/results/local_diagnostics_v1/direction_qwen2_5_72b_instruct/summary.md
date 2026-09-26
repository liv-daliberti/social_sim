# Direction companion: qwen2_5_72b_instruct

Exploratory companion to the existing numeric forecasting protocol.

Planned: 20 families, 240 independent direction responses.

| Context | Message | Target | Correct / planned | Unclear | Invalid or missing |
|---|---|---|---:|---:|---:|
| positive | new_news | increase | 12 / 20 | 0 | 0 |
| positive | no_news | unchanged | 18 / 20 | 0 | 0 |
| positive | repeated_news | unchanged | 19 / 20 | 0 | 0 |
| negative | new_news | decrease | 13 / 20 | 0 | 0 |
| negative | no_news | unchanged | 17 / 20 | 0 | 0 |
| negative | repeated_news | unchanged | 18 / 20 | 0 | 0 |
| broken | new_news | unchanged | 6 / 20 | 0 | 0 |
| broken | no_news | unchanged | 19 / 20 | 0 | 0 |
| broken | repeated_news | unchanged | 19 / 20 | 0 | 0 |
| masked | new_news | none | unscored | 0 | 0 |
| masked | no_news | unchanged | 20 / 20 | 0 | 0 |
| masked | repeated_news | unchanged | 20 / 20 | 0 | 0 |

Both directions correct: 6 / 20 families; 95% family-bootstrap interval: [0.1, 0.5].

All planned trials remain in scored denominators. Unclear, unchanged for signed targets, parse failures, and missing results fail. Masked new news has no target.

## Exploratory comparison with existing numeric updates

| | Numeric correct | Numeric incorrect or missing |
|---|---:|---:|
| Classification correct | 22 | 3 |
| Classification incorrect, unclear, or missing | 3 | 12 |

Comparison uses existing repeat-0 responses and strict positive/negative raw update signs. Numeric zero updates fail signed scoring. Endpoint flags and per-cell confusion counts are retained in summary.json.
