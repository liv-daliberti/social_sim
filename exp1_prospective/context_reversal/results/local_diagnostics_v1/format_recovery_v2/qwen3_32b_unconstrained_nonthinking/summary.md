# Context-reversal development pilot: qwen3_32b_unconstrained_nonthinking

Authored, unvalidated materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates.

Planned: 20 families, 80 context/repeat units, 320 response records.

## Coverage

| Context | Condition | Planned | Present | Valid probability | Missing record | Invalid present |
|---|---|---:|---:|---:|---:|---:|
| positive | baseline | 20 | 20 | 20 | 0 | 0 |
| positive | new_news | 20 | 20 | 20 | 0 | 0 |
| positive | no_news | 20 | 20 | 20 | 0 | 0 |
| positive | repeated_news | 20 | 20 | 20 | 0 | 0 |
| negative | baseline | 20 | 20 | 20 | 0 | 0 |
| negative | new_news | 20 | 20 | 20 | 0 | 0 |
| negative | no_news | 20 | 20 | 20 | 0 | 0 |
| negative | repeated_news | 20 | 20 | 20 | 0 | 0 |
| broken | baseline | 20 | 20 | 20 | 0 | 0 |
| broken | new_news | 20 | 20 | 20 | 0 | 0 |
| broken | no_news | 20 | 20 | 20 | 0 | 0 |
| broken | repeated_news | 20 | 20 | 20 | 0 | 0 |
| masked | baseline | 20 | 20 | 20 | 0 | 0 |
| masked | new_news | 20 | 20 | 20 | 0 | 0 |
| masked | no_news | 20 | 20 | 20 | 0 | 0 |
| masked | repeated_news | 20 | 20 | 20 | 0 | 0 |

## Planned-denominator outcomes

Zeros and invalid or missing trials count as failures. Paired reversal requires both signs to be correct.

| Outcome | Estimate [95% family interval] | Success / planned | Missing measurements |
|---|---|---:|---:|
| Primary raw: positive direction | 70.00% [50.00, 90.00] | 14 / 20 | 0 |
| Primary raw: negative direction | 55.00% [30.00, 75.00] | 11 / 20 | 0 |
| Primary raw: pooled direction | 62.50% [50.00, 75.00] | 25 / 40 | 0 |
| Primary raw: paired reversal | 30.00% [10.00, 50.00] | 6 / 20 | 0 |
| Drift adjusted: positive direction | 70.00% [50.00, 90.00] | 14 / 20 | 0 |
| Drift adjusted: negative direction | 55.00% [30.00, 75.00] | 11 / 20 | 0 |
| Drift adjusted: pooled direction | 62.50% [50.00, 75.00] | 25 / 40 | 0 |
| Drift adjusted: paired reversal | 30.00% [10.00, 50.00] | 6 / 20 | 0 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 41.90 [30.00, 53.80] | — | 20 / 20 |
| positive | new_news | 17.40 [6.00, 30.80] | 25.60 [17.60, 35.75] | 20 / 20 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| positive | repeated_news | 0.25 [0.00, 0.75] | 0.25 [0.00, 0.75] | 20 / 20 |
| positive | new_news minus no_news | 17.40 [6.00, 30.80] | 25.60 [17.60, 35.75] | 20 / 20 |
| positive | repeated_news minus no_news | 0.25 [0.00, 0.75] | 0.25 [0.00, 0.75] | 20 / 20 |
| negative | baseline probability | 42.00 [28.75, 55.25] | — | 20 / 20 |
| negative | new_news | -2.25 [-10.50, 7.75] | 16.75 [11.75, 22.25] | 20 / 20 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| negative | repeated_news | 0.00 [-0.75, 0.75] | 0.50 [0.00, 1.25] | 20 / 20 |
| negative | new_news minus no_news | -2.25 [-10.50, 7.75] | 16.75 [11.75, 22.25] | 20 / 20 |
| negative | repeated_news minus no_news | 0.00 [-0.75, 0.75] | 0.50 [0.00, 1.25] | 20 / 20 |
| broken | baseline probability | 29.95 [17.50, 42.40] | — | 20 / 20 |
| broken | new_news | 9.45 [1.25, 17.70] | 14.05 [7.55, 21.30] | 20 / 20 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| broken | repeated_news | 0.25 [0.00, 0.75] | 0.25 [0.00, 0.75] | 20 / 20 |
| broken | new_news minus no_news | 9.45 [1.25, 17.70] | 14.05 [7.55, 21.30] | 20 / 20 |
| broken | repeated_news minus no_news | 0.25 [0.00, 0.75] | 0.25 [0.00, 0.75] | 20 / 20 |
| masked | baseline probability | 37.50 [27.50, 45.00] | — | 20 / 20 |
| masked | new_news | 7.00 [-0.50, 15.00] | 15.50 [10.75, 21.25] | 20 / 20 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| masked | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| masked | new_news minus no_news | 7.00 [-0.50, 15.00] | 15.50 [10.75, 21.25] | 20 / 20 |
| masked | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **criterion_not_met** (complete_planned_observations); signed 90% interval [2.25, 16.25] pp, upper 95% absolute-movement bound 21.30 pp.
- drift_adjusted: **criterion_not_met** (complete_planned_observations); signed 90% interval [2.25, 16.25] pp, upper 95% absolute-movement bound 21.30 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
