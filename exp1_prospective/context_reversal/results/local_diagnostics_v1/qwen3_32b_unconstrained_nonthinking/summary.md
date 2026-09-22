# Context-reversal development pilot: qwen3_32b_unconstrained_nonthinking

Authored, unvalidated materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates.

Planned: 20 families, 80 context/repeat units, 320 response records.

## Coverage

| Context | Condition | Planned | Present | Valid probability | Missing record | Invalid present |
|---|---|---:|---:|---:|---:|---:|
| positive | baseline | 20 | 20 | 2 | 0 | 18 |
| positive | new_news | 20 | 20 | 1 | 0 | 19 |
| positive | no_news | 20 | 20 | 0 | 0 | 20 |
| positive | repeated_news | 20 | 20 | 0 | 0 | 20 |
| negative | baseline | 20 | 20 | 4 | 0 | 16 |
| negative | new_news | 20 | 20 | 0 | 0 | 20 |
| negative | no_news | 20 | 20 | 0 | 0 | 20 |
| negative | repeated_news | 20 | 20 | 1 | 0 | 19 |
| broken | baseline | 20 | 20 | 3 | 0 | 17 |
| broken | new_news | 20 | 20 | 0 | 0 | 20 |
| broken | no_news | 20 | 20 | 0 | 0 | 20 |
| broken | repeated_news | 20 | 20 | 0 | 0 | 20 |
| masked | baseline | 20 | 20 | 5 | 0 | 15 |
| masked | new_news | 20 | 20 | 1 | 0 | 19 |
| masked | no_news | 20 | 20 | 2 | 0 | 18 |
| masked | repeated_news | 20 | 20 | 0 | 0 | 20 |

## Planned-denominator outcomes

Zeros and invalid or missing trials count as failures. Paired reversal requires both signs to be correct.

| Outcome | Estimate [95% family interval] | Success / planned | Missing measurements |
|---|---|---:|---:|
| Primary raw: positive direction | 0.00% [interval unavailable] | 0 / 20 | 19 |
| Primary raw: negative direction | 0.00% [interval unavailable] | 0 / 20 | 20 |
| Primary raw: pooled direction | 0.00% [interval unavailable] | 0 / 40 | 39 |
| Primary raw: paired reversal | 0.00% [interval unavailable] | 0 / 20 | 20 |
| Drift adjusted: positive direction | 0.00% [interval unavailable] | 0 / 20 | 20 |
| Drift adjusted: negative direction | 0.00% [interval unavailable] | 0 / 20 | 20 |
| Drift adjusted: pooled direction | 0.00% [interval unavailable] | 0 / 40 | 40 |
| Drift adjusted: paired reversal | 0.00% [interval unavailable] | 0 / 20 | 20 |

## Baselines and movement

All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.

| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |
|---|---|---|---|---:|
| positive | baseline probability | 25.00 [interval unavailable] | — | 2 / 20 |
| positive | new_news | -20.00 [interval unavailable] | 20.00 [interval unavailable] | 1 / 20 |
| positive | no_news | unavailable | unavailable | 0 / 20 |
| positive | repeated_news | unavailable | unavailable | 0 / 20 |
| positive | new_news minus no_news | unavailable | unavailable | 0 / 20 |
| positive | repeated_news minus no_news | unavailable | unavailable | 0 / 20 |
| negative | baseline probability | 45.00 [interval unavailable] | — | 4 / 20 |
| negative | new_news | unavailable | unavailable | 0 / 20 |
| negative | no_news | unavailable | unavailable | 0 / 20 |
| negative | repeated_news | 0.00 [interval unavailable] | 0.00 [interval unavailable] | 1 / 20 |
| negative | new_news minus no_news | unavailable | unavailable | 0 / 20 |
| negative | repeated_news minus no_news | unavailable | unavailable | 0 / 20 |
| broken | baseline probability | 16.67 [interval unavailable] | — | 3 / 20 |
| broken | new_news | unavailable | unavailable | 0 / 20 |
| broken | no_news | unavailable | unavailable | 0 / 20 |
| broken | repeated_news | unavailable | unavailable | 0 / 20 |
| broken | new_news minus no_news | unavailable | unavailable | 0 / 20 |
| broken | repeated_news minus no_news | unavailable | unavailable | 0 / 20 |
| masked | baseline probability | 50.00 [interval unavailable] | — | 5 / 20 |
| masked | new_news | 10.00 [interval unavailable] | 10.00 [interval unavailable] | 1 / 20 |
| masked | no_news | 0.00 [interval unavailable] | 0.00 [interval unavailable] | 2 / 20 |
| masked | repeated_news | unavailable | unavailable | 0 / 20 |
| masked | new_news minus no_news | 10.00 [interval unavailable] | 10.00 [interval unavailable] | 1 / 20 |
| masked | repeated_news minus no_news | unavailable | unavailable | 0 / 20 |

## Broken-link stability

Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.

- raw_new_news: **indeterminate** (missing_planned_observations); signed 90% interval unavailable, upper 95% absolute-movement bound unavailable.
- drift_adjusted: **indeterminate** (missing_planned_observations); signed 90% interval unavailable, upper 95% absolute-movement bound unavailable.

Uncertainty: 2,000 whole-family bootstrap draws, seed 20260921; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.

Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.
