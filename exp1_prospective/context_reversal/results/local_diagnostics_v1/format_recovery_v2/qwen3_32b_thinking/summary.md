# Context-reversal development pilot: qwen3_32b_thinking

Authored, unvalidated materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates.

Planned: 20 families, 80 context/repeat units, 320 response records.

## Coverage

| Context | Condition | Planned | Present | Valid probability | Missing record | Invalid present |
|---|---|---:|---:|---:|---:|---:|
| positive | baseline | 20 | 20 | 18 | 0 | 2 |
| positive | new_news | 20 | 20 | 18 | 0 | 2 |
| positive | no_news | 20 | 20 | 18 | 0 | 2 |
| positive | repeated_news | 20 | 20 | 18 | 0 | 2 |
| negative | baseline | 20 | 20 | 19 | 0 | 1 |
| negative | new_news | 20 | 20 | 19 | 0 | 1 |
| negative | no_news | 20 | 20 | 19 | 0 | 1 |
| negative | repeated_news | 20 | 20 | 19 | 0 | 1 |
| broken | baseline | 20 | 20 | 19 | 0 | 1 |
| broken | new_news | 20 | 20 | 19 | 0 | 1 |
| broken | no_news | 20 | 20 | 19 | 0 | 1 |
| broken | repeated_news | 20 | 20 | 19 | 0 | 1 |
| masked | baseline | 20 | 20 | 19 | 0 | 1 |
| masked | new_news | 20 | 20 | 19 | 0 | 1 |
| masked | no_news | 20 | 20 | 19 | 0 | 1 |
| masked | repeated_news | 20 | 20 | 19 | 0 | 1 |

## Planned-denominator outcomes

Zeros and invalid or missing trials count as failures. Paired reversal requires both signs to be correct.

| Outcome | Estimate [95% family interval] | Success / planned | Missing measurements |
|---|---|---:|---:|
| Primary raw: positive direction | 80.00% [60.00, 95.00] | 16 / 20 | 2 |
| Primary raw: negative direction | 70.00% [50.00, 90.00] | 14 / 20 | 1 |
| Primary raw: pooled direction | 75.00% [60.00, 87.50] | 30 / 40 | 3 |
| Primary raw: paired reversal | 60.00% [35.00, 80.00] | 12 / 20 | 2 |
| Drift adjusted: positive direction | 80.00% [60.00, 95.00] | 16 / 20 | 2 |
| Drift adjusted: negative direction | 70.00% [50.00, 90.00] | 14 / 20 | 1 |
| Drift adjusted: pooled direction | 75.00% [60.00, 87.50] | 30 / 40 | 3 |
| Drift adjusted: paired reversal | 60.00% [35.00, 80.00] | 12 / 20 | 2 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 46.11 [36.32, 56.95] | — | 18 / 20 |
| positive | new_news | 18.33 [11.50, 24.74] | 21.11 [16.50, 26.11] | 18 / 20 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 18 / 20 |
| positive | repeated_news | 2.50 [0.00, 5.79] | 2.50 [0.00, 5.79] | 18 / 20 |
| positive | new_news minus no_news | 18.33 [11.50, 24.74] | 21.11 [16.50, 26.11] | 18 / 20 |
| positive | repeated_news minus no_news | 2.50 [0.00, 5.79] | 2.50 [0.00, 5.79] | 18 / 20 |
| negative | baseline probability | 54.17 [44.64, 64.97] | — | 19 / 20 |
| negative | new_news | -13.06 [-22.50, -1.77] | 23.52 [18.49, 29.30] | 19 / 20 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19 / 20 |
| negative | repeated_news | 0.53 [0.00, 1.67] | 0.53 [0.00, 1.67] | 19 / 20 |
| negative | new_news minus no_news | -13.06 [-22.50, -1.77] | 23.52 [18.49, 29.30] | 19 / 20 |
| negative | repeated_news minus no_news | 0.53 [0.00, 1.67] | 0.53 [0.00, 1.67] | 19 / 20 |
| broken | baseline probability | 48.95 [46.67, 50.00] | — | 19 / 20 |
| broken | new_news | 2.89 [-2.89, 8.75] | 7.63 [2.65, 13.16] | 19 / 20 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19 / 20 |
| broken | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19 / 20 |
| broken | new_news minus no_news | 2.89 [-2.89, 8.75] | 7.63 [2.65, 13.16] | 19 / 20 |
| broken | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19 / 20 |
| masked | baseline probability | 50.00 [50.00, 50.00] | — | 19 / 20 |
| masked | new_news | 5.26 [-4.41, 14.21] | 20.00 [16.25, 23.24] | 19 / 20 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19 / 20 |
| masked | repeated_news | 0.79 [0.00, 2.50] | 0.79 [0.00, 2.50] | 19 / 20 |
| masked | new_news minus no_news | 5.26 [-4.41, 14.21] | 20.00 [16.25, 23.24] | 19 / 20 |
| masked | repeated_news minus no_news | 0.79 [0.00, 2.50] | 0.79 [0.00, 2.50] | 19 / 20 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **indeterminate** (missing_planned_observations); signed 90% interval [-2.22, 7.89] pp, upper 95% absolute-movement bound 13.16 pp.
- drift_adjusted: **indeterminate** (missing_planned_observations); signed 90% interval [-2.22, 7.89] pp, upper 95% absolute-movement bound 13.16 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
