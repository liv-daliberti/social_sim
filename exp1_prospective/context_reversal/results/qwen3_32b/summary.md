# Context-reversal development pilot: qwen3_32b

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
| Primary raw: positive direction | 71.67% [55.00, 86.67] | 43 / 60 | 0 |
| Primary raw: negative direction | 56.67% [36.67, 76.67] | 34 / 60 | 0 |
| Primary raw: pooled direction | 64.17% [55.83, 73.33] | 77 / 120 | 0 |
| Primary raw: paired reversal | 31.67% [15.00, 48.33] | 19 / 60 | 0 |
| Drift adjusted: positive direction | 71.67% [55.00, 86.67] | 43 / 60 | 0 |
| Drift adjusted: negative direction | 56.67% [36.67, 76.67] | 34 / 60 | 0 |
| Drift adjusted: pooled direction | 64.17% [55.83, 73.33] | 77 / 120 | 0 |
| Drift adjusted: paired reversal | 31.67% [15.00, 48.33] | 19 / 60 | 0 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 35.50 [23.83, 46.67] | — | 60 / 60 |
| positive | new_news | 18.50 [8.83, 27.92] | 25.33 [20.08, 31.67] | 60 / 60 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| positive | repeated_news | 0.33 [0.00, 0.83] | 0.33 [0.00, 0.83] | 60 / 60 |
| positive | new_news minus no_news | 18.50 [8.83, 27.92] | 25.33 [20.08, 31.67] | 60 / 60 |
| positive | repeated_news minus no_news | 0.33 [0.00, 0.83] | 0.33 [0.00, 0.83] | 60 / 60 |
| negative | baseline probability | 44.18 [33.75, 53.95] | — | 60 / 60 |
| negative | new_news | -2.07 [-10.45, 7.75] | 19.73 [15.90, 24.50] | 60 / 60 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| negative | repeated_news | 1.67 [0.00, 4.67] | 1.67 [0.00, 4.67] | 60 / 60 |
| negative | new_news minus no_news | -2.07 [-10.45, 7.75] | 19.73 [15.90, 24.50] | 60 / 60 |
| negative | repeated_news minus no_news | 1.67 [0.00, 4.67] | 1.67 [0.00, 4.67] | 60 / 60 |
| broken | baseline probability | 33.15 [22.50, 43.78] | — | 60 / 60 |
| broken | new_news | 4.90 [-1.20, 11.08] | 13.10 [8.77, 17.52] | 60 / 60 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| broken | repeated_news | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| broken | new_news minus no_news | 4.90 [-1.20, 11.08] | 13.10 [8.77, 17.52] | 60 / 60 |
| broken | repeated_news minus no_news | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| masked | baseline probability | 44.75 [39.17, 50.08] | — | 60 / 60 |
| masked | new_news | 2.00 [-5.67, 9.75] | 16.33 [12.83, 19.67] | 60 / 60 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | repeated_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | new_news minus no_news | 2.00 [-5.67, 9.75] | 16.33 [12.83, 19.67] | 60 / 60 |
| masked | repeated_news minus no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **criterion_not_met** (complete_planned_observations); signed 90% interval [-0.35, 10.08] pp, upper 95% absolute-movement bound 17.52 pp.
- drift_adjusted: **criterion_not_met** (complete_planned_observations); signed 90% interval [-0.35, 10.08] pp, upper 95% absolute-movement bound 17.52 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
