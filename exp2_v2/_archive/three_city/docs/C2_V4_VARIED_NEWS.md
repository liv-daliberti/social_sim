# v4: does the finding survive when averaging is not an option?

## The worry v4 tests

Every v2 and v3 case carried the same `+8` news value. That made "average the
end-poll column" a *valid* estimator of the target city's response, and arguably
the obvious move on a table of eight near-identical experiments. The v3 finding —
that models land on the target-only estimate and ignore the two reference cities
— may therefore have been manufactured by that affordance rather than discovered.

The uniform value was not arbitrary: v2 introduced it to kill the temporal-state
ambiguity that broke v1, and to make the readout a single clean number. But it
may have made the target's own data feel self-sufficient in a way that no longer
tests anything.

## The single change

The displayed news values now vary, drawn from a fixed balanced multiset and
shuffled independently per city:

| | v3 | v4 |
|---|---|---|
| reference cases | `+8` × 6 | shuffle of `(+10, -10, +8, -8, +5, -5)` |
| target cases | `+8` × 8 | shuffle of `(+10, -10, +8, -8, +5, -5, +8, -8)` |
| forecast case | `+8` | `+8` (unchanged) |

Everything else is identical: gains 0.90 / 0.30, noise sd 4.30, cases start at
50.0, eight-case ladder, four background variants, exact factor balance.

**Why the forecast case stays fixed.** Holding it at `+8` keeps the readout
identical to v3 — implied response is still `(predicted_poll - 50) / 8`, the
decision boundary is still 0.60, and panel 3's ±0.30 targets are unchanged. So v3
and v4 numbers are directly comparable.

**Why the multisets sum to zero and are RMS-matched.** Summing to zero is what
breaks the shortcut: a column average lands near 50 and implies a response of
about **-0.017**, far outside the demonstrated band of 0.30 to 0.90. Matching the
root-mean-square to v3's uniform 8 (7.94 for references, 7.95 for the target)
preserves per-case information about the response, so difficulty is unchanged:

| ceiling p(correct) | k=1 | k=2 | k=3 | k=4 | k=8 |
|---|---|---|---|---|---|
| v3 | .639 | .734 | .775 | .826 | .921 |
| v4 | .630 | .686 | .761 | .800 | .918 |

Headroom over the target-only estimator is likewise comparable (v4: 1.23 at k=2,
0.78 at k=8).

## What this changes about the readout

Recovering the target's own response now requires a **slope**, not a mean. The
`frequentist` baseline already computed a regression through the origin, so it
needed no change and remains the correct information-matched "target only"
comparison. What changes is that a model can no longer reach it by averaging a
column.

Three gates pin this, and they would all fail on v3:

- `news_varies_within_each_city` — at least three distinct values per city;
- `column_average_implies_almost_no_response` — `|implied| < 0.15`;
- `column_average_falls_outside_demonstrated_band`.

## How to read the result

- **If models still sit on the target-only slope**, the v3 finding is not an
  averaging artefact. It becomes stronger: they compute the target-only
  estimator in whatever form the data demands, and still decline to combine it
  with the two demonstrated cities.
- **If they instead scatter, or fall back to 50**, then v3 was measuring the
  availability of a cheap shortcut, and the "does not use siblings as a prior"
  framing has to be withdrawn.
- **If they move toward the two-prototype forecaster**, the reference cities were
  being ignored only because the target looked self-sufficient.

A one-episode smoke test on all three models is suggestive of the first: the
column average was 50.42, the target-only slope 56.77, the truth 57.2, and the
three models answered 56.0, 56.0 and 55.0. They fitted the slope rather than
averaging, and still did not reach the truth. One episode proves nothing; the
48-episode run decides it.

## Status

- 4,320 frozen tasks, 40 design gates pass, 88 tests pass.
- Seed offset is 30,000, the same as v3. The noise realizations are nonetheless
  independent, because drawing and shuffling the news schedule consumes the
  generator before the gaussian errors: on the same seed, v3 and v4 share none of
  their eight target polls and their residuals are unrelated. Verified rather
  than assumed. The factor structure per episode index is unchanged, since
  `balanced_triplets` assigns response, label, and paraphrase family explicitly.
- v3 and its hint arm remain frozen and untouched; their scripts are not edited,
  because their validation reports record those scripts' sha256.
- The 48-episode run is in flight.
