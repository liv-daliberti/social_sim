# Four designs, and what each one actually established

Draft. v2, v3 and v4 are complete at 48 episodes; v5 is in flight (17 episodes at
time of writing) and every v5 number below is provisional.

This supersedes the framing in [FINDINGS.md](FINDINGS.md), which still describes
v2 alone and states the result more strongly than the later designs support.

## What changed, version to version

| | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| gains | 1.00 / 0.25 | 0.90 / 0.30 | 0.90 / 0.30 | 0.90 / 0.30 |
| noise sd | 3.25 | 4.30 | 4.30 | 4.30 |
| per-case `d'` | 1.85 | 1.12 | 1.12 | 1.12 |
| news per case | `+8` throughout | `+8` throughout | mixed, `+8` twice | mixed, no repeats |
| forecast case | `+8` | `+8` | `+8` | `+8`, absent from data |
| target-only estimator is | a column mean | a column mean | a column mean of 2 | a fitted slope |

Each step fixed a specific defect the previous one hid. None of them changed the
task the model sees beyond these knobs: the prompt, the ladder, the four
background variants and the balance are identical throughout.

## The headline metric across all four

Distance between a model's implied response and the **target-only estimator** —
the forecaster that uses City C's own cases and ignores the two examples. `0.00`
means the model *is* that estimator.

| arm | k | Claude | DeepSeek | GPT-5.4 |
|---|---|---|---|---|
| v2 | 2 | 0.008 | 0.043 | 0.003 |
| v2 | 8 | 0.008 | 0.062 | 0.011 |
| v3 | 2 | 0.020 | 0.096 | 0.003 |
| v3 | 8 | 0.009 | 0.082 | 0.006 |
| v4 | 2 | 0.330 | 0.401 | 0.288 |
| v4 | 8 | 0.190 | 0.248 | 0.228 |
| v5 *(17 eps)* | 2 | 0.228 | 0.240 | 0.227 |
| v5 *(17 eps)* | 8 | 0.167 | 0.309 | 0.145 |

**This is the finding, and it is not stable across designs.** In v2 and v3, where
the target-only estimator is a column average, the models *are* it to three
decimal places. In v4 and v5, where it requires a slope, they are an order of
magnitude further from it.

## What each version established

### v2 — the result, and the reason not to trust it yet

Models sat exactly on the column mean and left the whole ceiling gap unused. But
`d' = 1.85` saturated identification: the ceiling reached 0.917 accuracy by `k=2`
and 1.000 by `k=8`, so no model could visibly fall short of it, and "they
identify correctly" could not be distinguished from "the task is too easy to
fail".

### v3 — same result at honest difficulty, one claim withdrawn

Lowering `d'` to 1.12 graded the ladder (ceiling 0.792 at `k=2`, 0.938 at `k=8`)
and the headline survived unchanged. But it refuted a v2 claim that is still
written in FINDINGS.md: v2 reported models as approximately unbiased with spread
10–20× the ceiling's. At realistic difficulty the ceiling itself hedges, and the
excess is ~1.3×:

| arm | ceiling sd | model sd (C / D / G) |
|---|---|---|
| v2 | 0.374 | 0.39 / 0.37 / 0.39 |
| v3 | 0.275 | 0.35 / 0.35 / 0.34 |

**The "10–20×" figure in FINDINGS.md is an artefact of v2's over-easy
calibration and must be removed.**

### v4 — the finding was partly an artefact, and a new behaviour appeared

Varying the news broke column-averaging. Models did not fall back on the two
examples; they found a *different* shortcut. v4's target pool held `+8` twice
while the forecast was `+8`, so they subset to those two cases and averaged them.
GPT-5.4's forecasts landed within **0.019 poll points** of exactly that statistic.

That is a real result about how these models reach for the cheapest
sufficient-looking statistic, and it is worth reporting on its own. But it
disqualifies v4 as a test of the sibling-prior question.

### v5 — the clean test, in flight

Every news value appears once per city and the forecast value appears nowhere, so
matching returns nothing and a slope is the only route. Provisionally the models
sit *above* the target-only line rather than on it — worse than simply using City
C's own data — which is v4's signal reproducing without v4's flaw.

## What survives, and what does not

**Survives.** At `k=0`, with no City C cases to fit, all three models land on the
midpoint of the two demonstrated responses in every version. They do extract the
structure from two worked examples. And they do not treat those examples as a
prior once the target has any data of its own.

**Does not survive.** The clean statement "they compute the target-only estimator
and decline to shrink it" holds only when that estimator is a column average.
When it takes a regression they do not reliably compute it either — so the v2/v3
picture of a precise, deliberate estimator choice is too flattering.

**Was never supported.** That panel-1 accuracy demonstrates inference. The
plain target-only forecaster, which never reads City A or B, scores identically
to the ceiling in v2 and v3. This is already corrected in FINDINGS.md.

**Still open.** Whether the v5 gap reflects an inability to fit a slope from eight
noisy points, or a genuine refusal to use the examples. A slope-fitting control —
one city, no siblings, same regression — would separate these and does not exist
yet.

## Corrections owed to FINDINGS.md and PILOT_V2_AUDIT.md

1. Remove the "10–20× the ceiling's spread" claim (v3 refutes it; ~1.3×).
2. Remove "the means are 85–103% of ideal, nobody is hedging" — the ceiling
   hedges too once `d'` is realistic.
3. State that the headline holds only where the target-only estimator is an
   average, and report the v4/v5 divergence.
4. Record v4's match-and-average behaviour as a finding in its own right.
5. Mark every v2 number as calibration-specific rather than general.

## Provenance

Each version is frozen with its own engine, builder, validator, config, tests and
answer key. Earlier scripts are never edited, because each validation report
records their sha256. v2 40 gates, v3 37, v4 40, v5 42; 108 tests across all
versions. All four arms ran the same 48 balanced episodes with zero parse
failures, 4,032 prompts per arm.
