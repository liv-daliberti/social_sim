# Three-city C2 v8 five-round graded-similarity matched specification

This amendment was frozen on 2026-08-04 before any model call from this
similarity-informed design. The immediately preceding five-case midpoint-prior
run was stopped at user request and archived under
`full_k1_5_matched_midpoint_aborted_20260804`. Its partial responses are not
pooled with this run. All earlier partial designs remain separately archived.

Because earlier designs and partial responses were inspected before this
amendment, this is an amended full rerun rather than an untouched prospective
preregistration.

## Design objective

Create a large, defensible early advantage for combining A/B/C while ensuring
that both estimators target the same continuous City C response and become
nearly indistinguishable after sufficient City C evidence.

## Five evidence rounds

The reported horizontal coordinate is an evidence round, not a raw case count.

| Evidence round \(k\) | Completed City C cases |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |

All five rounds are measured. No unreported intermediate prefix is scored.

## Frozen world and graded similarity signal

- 120 episodes, seeds 80000--80119.
- Every polling row is an independent case beginning at 50.0 with net news
  \(+8\).
- Evaluator-only case noise has standard deviation 3.25.
- Cities A and B each show six representative cases.
- City C contains sixteen cases from the same seeded stream.
- A preregistered regional-profile index is displayed for every city.
- The high-response reference has index 100 and the low-response reference has
  index 0.
- City C index centers are 75 and 25 for the two balanced latent groups, with
  independent Gaussian index noise of 18 points and clipping to 0--100.
- The index is graded and genuinely overlapping: on the frozen seeds, 14 of 120
  City C indices cross the 50-point group midpoint.
- The index is generated from a separate seeded draw and does not alter polling
  outcomes.

The prompt describes the index as a descriptive composite, not an outcome or
response label. The strong arm says nearby indices tend to indicate related but
not necessarily identical responses. It never states that City C exactly
matches A or B.

## Prompt arms

- **A -- structure-blind:** displays all records and indices without a structural
  instruction.
- **B -- relevance hint:** asks whether records, backgrounds, and indices are
  informative.
- **C -- continuous-pooling hint:** directs the forecaster to form a continuous
  initial estimate from A/B and increase the weight on C as evidence grows.

All other prompt content and task data are identical across arms.

## Matched continuous estimators

- **Use City C only:** through-origin response regression using the cumulative C
  cases available in the current evidence round.
- **Combine A/B/C (matched):** interpolate a continuous prior mean between the
  displayed A/B slopes using the displayed indices. Prior variance is the
  variance of a uniform distribution spanning those slopes,
  \((\hat g_A-\hat g_B)^2/12\). Displayed A/B residuals estimate case noise.
  The likelihood uses exactly the same C regression as the C-only estimator.

Both estimate the same continuous City C coefficient. A/B supplies
regularization, not an exact-classification shortcut. Mean A/B weight falls from
0.753 in round 1 to 0.176 in round 5.

Frozen-seed baseline diagnostics are:

| Round | C cases | C-only MAE | Matched A/B/C MAE | Gap |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2.788 | 1.359 | 1.429 |
| 2 | 2 | 2.029 | 1.275 | 0.755 |
| 3 | 4 | 1.336 | 1.043 | 0.293 |
| 4 | 8 | 0.838 | 0.782 | 0.056 |
| 5 | 16 | 0.607 | 0.580 | 0.026 |

Thus the initial gap is large, while the endpoint is visually and numerically
shared. Exact-type and context-aware evaluator oracles remain diagnostic
ceilings only and are not labeled as the matched A/B/C baseline.

## Execution and size

Models are Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4. All nine model-by-arm
shards run concurrently on the login node. No Slurm job is used. Artifacts are
stored under `data/three_city_c2_v8/full_k1_5_similarity`.

- Each model-arm has \(120\times4\times5=2{,}400\) calls.
- Each model has 7,200 calls.
- The complete run has 21,600 calls.
- Uncertainty intervals use deterministic 10,000-draw episode bootstraps.
- A received but unparseable completion is terminal. A pre-completion transport
  failure remains retryable on a later invocation.
