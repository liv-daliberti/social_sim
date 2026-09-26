# Context-reversal development pilot: qwen2_5_72b_instruct

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
| Primary raw: positive direction | 73.33% [53.33, 90.00] | 44 / 60 | 0 |
| Primary raw: negative direction | 51.67% [30.00, 71.71] | 31 / 60 | 0 |
| Primary raw: pooled direction | 62.50% [52.48, 73.33] | 75 / 120 | 0 |
| Primary raw: paired reversal | 30.00% [11.67, 50.00] | 18 / 60 | 0 |
| Drift adjusted: positive direction | 73.33% [53.33, 90.00] | 44 / 60 | 0 |
| Drift adjusted: negative direction | 51.67% [30.00, 71.71] | 31 / 60 | 0 |
| Drift adjusted: pooled direction | 62.50% [52.48, 73.33] | 75 / 120 | 0 |
| Drift adjusted: paired reversal | 30.00% [11.67, 50.00] | 18 / 60 | 0 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 60.75 [53.50, 67.84] | — | 60 / 60 |
| positive | new_news | 6.58 [-1.17, 13.42] | 17.75 [15.17, 20.50] | 60 / 60 |
| positive | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| positive | repeated_news | 0.87 [0.03, 2.25] | 1.03 [0.03, 2.45] | 60 / 60 |
| positive | new_news minus no_news | 6.58 [-1.17, 13.42] | 17.75 [15.17, 20.50] | 60 / 60 |
| positive | repeated_news minus no_news | 0.87 [0.03, 2.25] | 1.03 [0.03, 2.45] | 60 / 60 |
| negative | baseline probability | 59.75 [52.92, 66.42] | — | 60 / 60 |
| negative | new_news | -2.87 [-11.71, 5.75] | 19.38 [16.33, 22.50] | 60 / 60 |
| negative | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| negative | repeated_news | 0.28 [0.00, 0.70] | 0.28 [0.00, 0.70] | 60 / 60 |
| negative | new_news minus no_news | -2.87 [-11.71, 5.75] | 19.38 [16.33, 22.50] | 60 / 60 |
| negative | repeated_news minus no_news | 0.28 [0.00, 0.70] | 0.28 [0.00, 0.70] | 60 / 60 |
| broken | baseline probability | 54.17 [48.25, 60.08] | — | 60 / 60 |
| broken | new_news | -1.50 [-8.75, 5.17] | 13.33 [9.17, 17.42] | 60 / 60 |
| broken | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| broken | repeated_news | 0.02 [0.00, 0.05] | 0.02 [0.00, 0.05] | 60 / 60 |
| broken | new_news minus no_news | -1.50 [-8.75, 5.17] | 13.33 [9.17, 17.42] | 60 / 60 |
| broken | repeated_news minus no_news | 0.02 [0.00, 0.05] | 0.02 [0.00, 0.05] | 60 / 60 |
| masked | baseline probability | 52.58 [50.17, 56.42] | — | 60 / 60 |
| masked | new_news | -1.08 [-8.75, 6.17] | 16.42 [13.92, 18.92] | 60 / 60 |
| masked | no_news | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| masked | repeated_news | 0.17 [0.00, 0.50] | 0.17 [0.00, 0.50] | 60 / 60 |
| masked | new_news minus no_news | -1.08 [-8.75, 6.17] | 16.42 [13.92, 18.92] | 60 / 60 |
| masked | repeated_news minus no_news | 0.17 [0.00, 0.50] | 0.17 [0.00, 0.50] | 60 / 60 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **criterion_not_met** (complete_planned_observations); signed 90% interval [-7.67, 3.92] pp, upper 95% absolute-movement bound 17.42 pp.
- drift_adjusted: **criterion_not_met** (complete_planned_observations); signed 90% interval [-7.67, 3.92] pp, upper 95% absolute-movement bound 17.42 pp.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
