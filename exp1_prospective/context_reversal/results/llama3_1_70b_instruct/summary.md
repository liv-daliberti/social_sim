# Context-reversal development pilot: llama3_1_70b_instruct

Authored, unvalidated materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates.

Planned: 20 families, 240 context/repeat units, 960 response records.

## Coverage

| Context | Condition | Planned | Present | Valid probability | Missing record | Invalid present |
|---|---|---:|---:|---:|---:|---:|
| positive | baseline | 60 | 60 | 60 | 0 | 0 |
| positive | new_news | 60 | 60 | 60 | 0 | 0 |
| positive | no_news | 60 | 60 | 60 | 0 | 0 |
| positive | repeated_news | 60 | 60 | 60 | 0 | 0 |
| negative | baseline | 60 | 60 | 60 | 0 | 0 |
| negative | new_news | 60 | 60 | 60 | 0 | 0 |
| negative | no_news | 60 | 60 | 60 | 0 | 0 |
| negative | repeated_news | 60 | 60 | 60 | 0 | 0 |
| broken | baseline | 60 | 60 | 60 | 0 | 0 |
| broken | new_news | 60 | 60 | 60 | 0 | 0 |
| broken | no_news | 60 | 60 | 60 | 0 | 0 |
| broken | repeated_news | 60 | 60 | 60 | 0 | 0 |
| masked | baseline | 60 | 60 | 60 | 0 | 0 |
| masked | new_news | 60 | 60 | 60 | 0 | 0 |
| masked | no_news | 60 | 60 | 60 | 0 | 0 |
| masked | repeated_news | 60 | 60 | 60 | 0 | 0 |

## Planned-denominator outcomes

Zeros and invalid or missing trials count as failures. Paired reversal requires both signs to be correct.

| Outcome | Estimate [95% family interval] | Success / planned | Missing measurements |
|---|---|---:|---:|
| Primary raw: positive direction | 76.67% [58.33, 91.67] | 46 / 60 | 0 |
| Primary raw: negative direction | 53.33% [31.67, 75.00] | 32 / 60 | 0 |
| Primary raw: pooled direction | 65.00% [53.33, 77.50] | 78 / 120 | 0 |
| Primary raw: paired reversal | 36.67% [16.67, 56.67] | 22 / 60 | 0 |
| Drift adjusted: positive direction | 76.67% [58.33, 91.67] | 46 / 60 | 0 |
| Drift adjusted: negative direction | 53.33% [31.67, 75.00] | 32 / 60 | 0 |
| Drift adjusted: pooled direction | 65.00% [53.33, 77.50] | 78 / 120 | 0 |
| Drift adjusted: paired reversal | 36.67% [16.67, 56.67] | 22 / 60 | 0 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 38.96 [28.33, 49.58] | — | 60 / 60 |
| positive | new_news | 14.71 [2.67, 24.79] | 25.38 [19.79, 31.46] | 60 / 60 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| positive | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| positive | new_news minus no_news | 14.71 [2.67, 24.79] | 25.38 [19.79, 31.46] | 60 / 60 |
| positive | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| negative | baseline probability | 42.68 [30.20, 54.36] | — | 60 / 60 |
| negative | new_news | -3.44 [-14.43, 7.70] | 21.17 [14.10, 29.31] | 60 / 60 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| negative | repeated_news | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| negative | new_news minus no_news | -3.44 [-14.43, 7.70] | 21.17 [14.10, 29.31] | 60 / 60 |
| negative | repeated_news minus no_news | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| broken | baseline probability | 36.67 [25.83, 48.33] | — | 60 / 60 |
| broken | new_news | 2.33 [-5.84, 9.34] | 10.50 [4.75, 17.75] | 60 / 60 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| broken | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| broken | new_news minus no_news | 2.33 [-5.84, 9.34] | 10.50 [4.75, 17.75] | 60 / 60 |
| broken | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | baseline probability | 47.73 [40.83, 54.03] | — | 60 / 60 |
| masked | new_news | 3.88 [-5.19, 12.90] | 16.71 [11.85, 22.40] | 60 / 60 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | new_news minus no_news | 3.88 [-5.19, 12.90] | 16.71 [11.85, 22.40] | 60 / 60 |
| masked | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **criterion_not_met** (complete_planned_observations); signed 90% interval [-4.34, 8.50] pp, upper 95% absolute-movement bound 17.75 pp.
- drift_adjusted: **criterion_not_met** (complete_planned_observations); signed 90% interval [-4.34, 8.50] pp, upper 95% absolute-movement bound 17.75 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
