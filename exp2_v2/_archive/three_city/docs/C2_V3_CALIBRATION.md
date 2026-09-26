# Why v3 recalibrates the gain and noise

v2's task structure is unchanged. Only three generative constants move, plus the
design gates that were unknowingly tuned to v2's difficulty.

| | v2 | v3 |
|---|---|---|
| responsive gain | 1.00 | 0.90 |
| buffered gain | 0.25 | 0.30 |
| case noise (sd) | 3.25 | 4.30 |
| poll gap between kinds | 6.0 pts | 4.8 pts |
| gain ratio | 4.0× | 3.0× |
| per-case discriminability `d' = gap / sd` | **1.85** | **1.12** |

## The problem v3 fixes

v2's `d'` was too high, so identification saturated:

| ceiling p(correct) | k=0 | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 |
|---|---|---|---|---|---|---|---|
| v2 | .500 | .742 | .847 | **.934** | .964 | .988 | **.997** |
| v3 | .500 | .639 | .734 | .775 | .826 | .876 | **.921** |

In v2 the problem was effectively solved by `k=3`; the upper half of the ladder
added 0.063. Five of the nine rungs carried no information, and — the real cost —
**no model could visibly fall short of the ceiling**, so "the models identify the
city correctly" could not be distinguished from "the task is too easy to fail".
All three models sat at 1.000 from `k=4`.

In v3 the upper half adds 0.146 and the ladder top is informative without being
settled, so a forecaster that under-uses evidence now has room to show it.

## Why noise, not a narrower gain gap

Both knobs move `d'`, but they trade differently against effect size. Difficulty
scales as `gap / sd`; the accuracy headroom over a plain target average scales as
`sd`. Simulated at 40,000 draws per rung:

| setting | `d'` | accuracy k=1→8 | headroom at k≥3 |
|---|---|---|---|
| v2: 1.00/0.25, sd 3.25 | 1.85 | .82 → .995 | ~0.95 |
| narrow the gap: 0.85/0.40, sd 3.25 | 1.11 | .71 → .94 | ~0.61 |
| widen the noise: 1.00/0.25, sd 5.40 | 1.11 | .71 → .94 | ~1.01 |
| **v3: 0.90/0.30, sd 4.30** | **1.12** | **.71 → .94** | **~0.83** |

Narrowing the gain gap grades the curve at the cost of roughly 40% of the effect
size the experiment exists to measure. Widening the noise grades it for free. v3
takes most of the noise route and only a little of the gap route, because two
properties of v2's gains were independently worth dropping:

- `1.00` is exact pass-through — an `+8` news event moving the poll exactly 8
  points, which is a strange thing to posit as one of two modal types;
- `4×` is a caricatured ratio between "responsive" and "buffered".

`0.90 / 0.30` gives 3×, and what a reader sees is responsive cities at 57.2 ± 4.3
against buffered at 52.4 ± 4.3 — overlapping and noisy, which is what small local
polls actually look like.

## Gates retuned, and one added

Three gates encoded v2's saturation and were retuned rather than dropped:

- `one_case_is_informative_not_decisive`: `0.65 < p(1) < 0.85` → `0.55 … 0.75`
- `ladder_top_overrides_misleading_context`: `> 0.95` → `> 0.78`
- `ladder_top_is_strongly_informative` (`p(8) > 0.98`) was **replaced**, because
  it asserted the very thing that made v2 uninformative.

Two gates now encode the v3 intent directly:

- `ladder_top_is_informative_but_not_saturated`: `0.85 < p(8) < 0.96`
- `upper_half_of_ladder_still_carries_information`: `p(8) − p(3) ≥ 0.10`

Checked against both designs: v2 gives an upper-half gain of 0.063 and **fails**;
v3 gives 0.146 and passes. Had these gates existed, v2's saturation would have
been caught before any model ran.

## Status

- v2 is frozen and untouched. Its scripts are not edited, because its validation
  report records their sha256 — hence the duplicated `*_v3_*` modules rather than
  a shared parameterised one.
- v2's finding stands on its own; v3 tests whether it survives when
  identification is genuinely hard. It might not, which is the point of running
  it.
- v3 uses seed offset 30,000 against v2's 10,000, so the noise draws are
  independent. The factor structure per episode index is unchanged, since
  `balanced_triplets` assigns response, label, and paraphrase family explicitly
  rather than from the seed.
- 37 design gates pass; 69 tests pass.
