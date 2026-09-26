# Preregistration: natural continuous-profile three-city experiment (C2 v9)

## Motivation

This version replaces the v8 prompt-only manipulation. In v8, every arm saw
Cities A and B plus conspicuous 0/100 profile anchors, so an uninstructed model
could already pool. V9 directly separates the value of cross-city information
from the incremental value of a non-prescriptive graded relevance hint.

The paused v8 collection remains archived at
`data/three_city_c2_v8/full_k1_5_similarity/` and is not combined with v9.

## Confirmatory arms

1. **City C evidence only (`c_only`)**: City C's completed observations and
   continuous profile score are shown. No City A or B record is shown.
2. **A/B/C information (`abc`)**: the same City C material plus City A and B
   records and continuous profile scores. Instructions are neutral.
3. **A/B/C plus graded relevance hint (`abc_relevance`)**: information is
   byte-for-byte identical to `abc`; one frozen sentence says that closer
   profile scores are generally associated with more similar responses while
   substantial city-specific variation remains. It does not prescribe an
   estimator or an evidence-weighting rule.

The primary contrast is `abc - c_only`, which estimates the value of access to
cross-city information. The secondary contrast is `abc_relevance - abc`, which
estimates the incremental effect of explaining score relevance. V9 does not claim that
all three prompts contain identical information.

## Evidence rounds

| Evidence round | Completed City C cases |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |

Measurements are taken at every evidence round.

## Continuous data-generating process

There are no discrete city types or class labels.

For each of 120 frozen episodes:

- a lower reference score is sampled uniformly from `[0.10, 0.35]`;
- a higher reference score is sampled uniformly from `[0.65, 0.90]`;
- City C's score is sampled continuously between the two reference scores,
  remaining at least 0.05 inside each endpoint;
- which visible label, A or B, receives the higher score is balanced 60/60;
- an episode intercept is sampled uniformly from `[0.30, 0.45]`;
- an episode profile slope is sampled uniformly from `[0.35, 0.65]`;
- each city's latent response receives an independent Gaussian deviation with
  SD `0.18` response units;
- each completed polling case receives independent Gaussian noise with SD
  `4.0` poll points.

For city `j` with normalized profile score `s_j`, its latent response is

`theta_j = alpha + beta * s_j + eta_j`, where `eta_j ~ Normal(0, 0.18^2)`.

Every displayed case begins at 50.0 and has net news +8. Its displayed ending
poll is the latent expected poll plus case noise, rounded to one decimal place.
The private gold forecast is `50 + 8 * theta_C`.

Reference scores are never fixed at 0 or 100. City-specific deviations ensure
that profile proximity is informative but not deterministic.

## Matched estimator benchmarks

The City C benchmark fits a through-origin response regression using only the
displayed City C cases.

The A/B/C benchmark:

1. fits the same response regression separately to the eight A cases and eight
   B cases;
2. linearly interpolates those fitted slopes at City C's continuous score;
3. derives predictive prior variance from the preregistered city-deviation
   variance plus A/B sampling variance; and
4. updates that prior with exactly the same City C likelihood used by the
   City-C-only estimator.

Consequently, the estimators differ strongly when City C evidence is sparse
and converge as City C evidence accumulates.

Before model collection, the frozen 120-episode projection must satisfy:

- round-1 A/B/C advantage greater than 1.50 MAE points;
- round-2 advantage greater than 0.75 MAE points;
- A/B/C lower MAE in rounds 1–4;
- absolute estimator gap below 0.10 at round 5;
- both round-5 MAEs below 0.90;
- A/B reference weight above 0.70 at round 1 and below 0.25 at round 5.

These gates are evaluated before any model calls and are not tuned against
model outputs.

## Collection and reporting

The frozen seed offset is 91,000. There are 120 episodes × 5 rounds = 600 tasks
per arm, 1,800 calls per model, and 5,400 calls across Claude Opus 4.8,
DeepSeek V4 Pro, and GPT-5.4.

The figure reports MAE at rounds 1–5 for every model and arm, with the matched
City-C-only and A/B/C estimator curves overlaid. All incomplete figures are
marked interim and non-confirmatory.
