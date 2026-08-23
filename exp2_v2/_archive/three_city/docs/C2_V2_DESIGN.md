# C2 v2: minimal independent-case design

## Why v1's frequentist looked unexpectedly strong

The v1 target schedule used a small first shock followed by a large `+12`
shock. After the second observation, an unregularized target-only regression
had one very high-leverage measurement and could estimate the response
coefficient unusually well. Its 0.91-poll-point pilot MAE at `k=2` was a
consequence of the information supplied, not a coding error.

The same schedule also left a separate ambiguity: alternating reference shocks
kept returning polls near 50, so a model could not tell cleanly whether the next
poll continued from the current poll or reset around the baseline.

## Minimal v2 task

V2 removes both complications.

- Every row is an independent case, explicitly not a time series.
- Every case starts at 50.0.
- Every reference, target, and test case uses the same `+8` news value.
- City A and City B each show six repeated cases.
- City C reveals zero to eight repeated cases; curves report the ladder
  `0, 1, 2, 3, 4, 6, 8`.
- The model predicts one new City C end-of-case poll.

The two reference samples are constructed to have mean-zero residuals, so their
sample averages clearly demonstrate their respective behavior while retaining
within-city noise. City C cases remain ordinary independent noisy draws.

The model is still not told that there are two types, that City C corresponds to
A or B, that a response parameter exists, or what any prior or noise level is.
Only the non-latent measurement protocol is made explicit.

## Fair frequentist comparison

The target-only frequentist is just the City C sample mean expressed as a
forecast: unavailable at `k=0`, then the average of the `k` completed City C
results. Across the 120 frozen episodes:

| City C cases | Frequentist MAE | Pooled references | Bayes ceiling | Gap to ceiling |
|---:|---:|---:|---:|---:|
| 0 | unavailable | 3.00 | 3.00 | — |
| 1 | 3.09 | 3.00 | 1.55 | 1.54 |
| 2 | 1.97 | 3.00 | 0.92 | 1.06 |
| 3 | 1.40 | 3.00 | 0.40 | 1.01 |
| 4 | 1.19 | 3.00 | 0.21 | 0.98 |
| 6 | 0.95 | 3.00 | 0.07 | 0.88 |
| 8 | 0.91 | 3.00 | 0.02 | 0.89 |

The frequentist improves normally with repeated observations and is not
artificially weakened. The ladder spans the regime the experiment is about: at
`k=0` only the demonstrated cities can help, and by `k=8` City C's own record
carries most of the information.

The gap to the ceiling narrows but does not close, and that is structural rather
than a tuning choice. The ceiling holds a two-point prior, so it identifies the
city outright and its error collapses exponentially; a sample mean only improves
as `1/sqrt(k)`. Closing the gap to roughly 0.3 points would take about 64 cases,
so `TARGET_CASES = 8` is an interim anchor.

## Context and override

The relevant and orthogonal target descriptions are now tightly length-matched
(16.67 versus 17.33 mean words). All numerical content is identical across the
four paired variants.

The hidden contextual oracle's probability on the correct pattern evolves as:

| City C cases | No context | Misleading context |
|---:|---:|---:|
| 0 | 0.50 | 0.20 |
| 1 | 0.74 | 0.56 |
| 2 | 0.85 | — |
| 3 | 0.93 | — |
| 8 | 1.00 | 0.99 |

Context matters when City C has little evidence, and repeated observations
eventually override a misleading cue outright.

This also fixes the benchmark for the model comparison. A rational forecaster
holding the true 80% cue reliability *is* helped by an aligned background and
*is* hurt by a misleading one, so comparing model movement to zero would penalise
correct behavior. Context contrasts are therefore reported as paired changes
against the same episode's no-context prompt, minus the oracle's own paired
change. Results are in [PILOT_V2_AUDIT.md](PILOT_V2_AUDIT.md).
