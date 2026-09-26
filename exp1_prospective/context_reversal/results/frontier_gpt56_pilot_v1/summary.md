# Context-reversal development pilot: gpt_5_6_frontier

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
| Primary raw: positive direction | 95.00% [85.00, 100.00] | 19 / 20 | 0 |
| Primary raw: negative direction | 90.00% [75.00, 100.00] | 18 / 20 | 0 |
| Primary raw: pooled direction | 92.50% [85.00, 100.00] | 37 / 40 | 0 |
| Primary raw: paired reversal | 85.00% [70.00, 100.00] | 17 / 20 | 0 |
| Drift adjusted: positive direction | 95.00% [85.00, 100.00] | 19 / 20 | 0 |
| Drift adjusted: negative direction | 90.00% [75.00, 100.00] | 18 / 20 | 0 |
| Drift adjusted: pooled direction | 92.50% [85.00, 100.00] | 37 / 40 | 0 |
| Drift adjusted: paired reversal | 85.00% [70.00, 100.00] | 17 / 20 | 0 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 45.70 [39.40, 52.40] | — | 20 / 20 |
| positive | new_news | 9.60 [6.50, 12.10] | 11.10 [9.55, 12.60] | 20 / 20 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| positive | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| positive | new_news minus no_news | 9.60 [6.50, 12.10] | 11.10 [9.55, 12.60] | 20 / 20 |
| positive | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| negative | baseline probability | 49.15 [44.05, 54.30] | — | 20 / 20 |
| negative | new_news | -10.20 [-13.50, -6.40] | 12.10 [9.90, 14.30] | 20 / 20 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| negative | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| negative | new_news minus no_news | -10.20 [-13.50, -6.40] | 12.10 [9.90, 14.30] | 20 / 20 |
| negative | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| broken | baseline probability | 46.50 [42.75, 49.75] | — | 20 / 20 |
| broken | new_news | -0.50 [-1.50, 0.00] | 0.50 [0.00, 1.50] | 20 / 20 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| broken | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| broken | new_news minus no_news | -0.50 [-1.50, 0.00] | 0.50 [0.00, 1.50] | 20 / 20 |
| broken | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| masked | baseline probability | 48.35 [46.00, 50.05] | — | 20 / 20 |
| masked | new_news | -0.45 [-3.65, 2.80] | 5.15 [2.75, 7.70] | 20 / 20 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| masked | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |
| masked | new_news minus no_news | -0.45 [-3.65, 2.80] | 5.15 [2.75, 7.70] | 20 / 20 |
| masked | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 20 / 20 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **criterion_met** (complete_planned_observations); signed 90% interval [-1.50, 0.00] pp, upper 95% absolute-movement bound 1.50 pp.
- drift_adjusted: **criterion_met** (complete_planned_observations); signed 90% interval [-1.50, 0.00] pp, upper 95% absolute-movement bound 1.50 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
