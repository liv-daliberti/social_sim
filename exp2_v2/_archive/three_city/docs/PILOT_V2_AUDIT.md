# Three-city C2 v2 pilot audit

## Decision

**The v2 design is sound and cleared to scale.** Both defects the v1 pilot
exposed are fixed. The extended City C ladder, the pruned baseline set, and the
rational-agent benchmark for context effects together produced a clean reading of
what the models are and are not doing. One residual issue is attributable to a
model rather than to the design.

Sample: episodes 0–47 of the frozen `tasks_c2_v2.jsonl`, all four context
variants, prefixes `k = 0, 1, 2, 3, 4, 6, 8`. That is **1,344 prompts per model
and 4,032 total**. Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4 each completed
all 1,344 with zero parse failures, on byte-identical prompts.

Episodes 0–47 are four complete 12-episode blocks of the generator's balance
schedule, so target response, the A/B label, and the context paraphrase family
are all exactly crossed: 24/24 on type, 24/24 on which reference the target
matches, 16/16/16 across families, and all 12 factor cells at n=4. Episodes
48–119 of the frozen file are deliberately unrun so a confirmatory sample still
has held-out episodes; these 48 must not be pooled into that estimate.

## Design changes in this round

**City C's ladder runs to eight cases** (`TARGET_CASES = 8`), reported at the
rungs `0, 1, 2, 3, 4, 6, 8`. Adjacent full-ladder rungs differ by less than the
sampling error at 120 episodes, so curves and design gates use these rungs.

**The baseline set is now two information-matched anchors plus one ceiling:**

- **Frequentist target-only** — City C's completed cases, nothing else.
- **Pooled exemplar** — the two demonstrated cities, nothing else. Flat in `k`.
- **Bayes ceiling** — additionally knows the private generative process.

Naive, two-prototype, and clairvoyant were removed: the first two were
respectively uninformative and hand-built for this exact task, and the
clairvoyant prediction is simply the scoring target restated. The two survivors
bracket the model from opposite directions — use only the target, or use only the
references — which is the contrast the experiment is about.

### The ladder narrows the ceiling gap without closing it

This is structural, not a tuning problem. The ceiling holds a two-point prior, so
it identifies the type outright and its error collapses exponentially. The
frequentist estimates a continuous mean, so it only improves as `1/sqrt(k)`.
Over 20,000 simulated draws:

| k | frequentist | Bayes ceiling | gap |
|---:|---:|---:|---:|
| 1 | 2.58 | 1.51 | 1.07 |
| 4 | 1.28 | 0.29 | 0.99 |
| 8 | 0.92 | 0.04 | 0.88 |
| 16 | 0.64 | 0.001 | 0.64 |
| 64 | 0.32 | 0.000 | 0.32 |

`TARGET_CASES = 8` is therefore an **interim** anchor: the frequentist still
trails the ceiling by about 0.9 poll points there. Bringing the gap to roughly
0.3 points — about 10% of the 3.0-point error at `k=0` — needs about 64 cases.
`test_ceiling_gap_narrows_but_stays_open_at_the_ladder_top` pins this so a later
change to the anchor is a deliberate edit rather than silent drift.

Figures shade the region between the ceiling and the frequentist in green: it is
the headroom that the demonstrated cities plus the latent structure make
available over averaging City C alone.

## The two v1 defects are fixed

### 1. The temporal ambiguity is gone

Every v2 case starts at the same known poll, so the readout is a single implied
response, `(predicted_poll - 50) / net_news`. A forecast can no longer be right
about the pattern and wrong about the state transition. The v1 signature of that
confound was Claude scoring perfect pattern accuracy at `k=2` while carrying
5.525-point forecast MAE. In v2 the two move together for every model:

| Model | MAE k=0 | MAE k=4 | MAE k=8 | Accuracy k=0 | Accuracy k=8 |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.8 | 2.946 | 0.998 | 0.885 | 0.542 | 1.000 |
| DeepSeek V4 Pro | 3.108 | 0.902 | 0.794 | 0.438 | 0.958 |
| GPT-5.4 | 2.994 | 0.996 | 0.867 | 0.490 | 1.000 |

MAE falls monotonically in `k` for all three and accuracy rises to 1.000 (0.958
for DeepSeek) by `k=4`. Accuracy at `k=0` is chance by construction.

**A limit on what the accuracy column shows.** It is matched exactly by the plain
average of City C's own cases (0.917 at `k=2`, 1.000 at `k=8`), which never reads
the reference cities. So the accuracy curve measures whether a forecast falls on
the correct side of the boundary, not whether the two-city structure was used to
put it there. The evidence for structure use is the `k=0` column, where there is
nothing to average and all three land on the 0.625 midpoint of the two
demonstrations (0.605-0.619), plus the unscored rationales. See
[FINDINGS.md](FINDINGS.md).

### 2. Misleading-context override now works for every model

Measured against the rational benchmark below, Claude and GPT converge fully.
Claude's v1 failure — error *rising* from 0.740 to 1.203 — does not recur.

## Context use, measured against a rational agent

The earlier framing was unfair to the models. A rational forecaster holding the
evaluator's true 80% cue reliability *is* helped by an aligned background and *is*
hurt by a misleading one, so comparing model movement to zero penalises correct
behavior. Every context contrast is now a **paired** change against the same
episode's no-context prompt, benchmarked against the Bayes oracle's own paired
change, with 2.5–97.5 percentile bootstrap intervals over 48 episodes.

The qualitative pattern is right for all three: aligned backgrounds reduce
response error, misleading backgrounds raise it, and both effects shrink as City
C's cases accumulate — the shape the oracle also shows.

### How the background readouts are constructed

Three independent questions, each reported as **levels** for the models and for
the rational benchmark. Nothing subtracts one agent from another — see the
withdrawn framings below for why that matters.

1. **Magnitude** — how far adding one background moves the implied response,
   against the same episode's no-context prompt.
2. **Specificity** — the same measurement using an equally detailed but
   mechanism-irrelevant background. Should be zero.
3. **Direction** — the fraction of episodes where the high-transmission wording
   yields a higher implied response than the buffered wording. Direction only,
   independent of magnitude.

### The models weight the wording more heavily than its reliability warrants

Movement magnitude, no-context baseline, against the oracle holding the
evaluator's true 80% cue reliability:

| Forecaster | k=0 | k=2 | k=8 |
|---|---:|---:|---:|
| Claude Opus 4.8 | 0.378 [0.373, 0.383] | 0.179 [0.138, 0.221] | 0.005 |
| DeepSeek V4 Pro | 0.381 [0.375, 0.389] | 0.317 [0.264, 0.373] | 0.159 |
| GPT-5.4 | 0.338 [0.320, 0.355] | 0.027 [0.006, 0.055] | 0.009 |
| Bayes oracle | 0.225 | 0.075 | 0.002 |

All three move roughly 1.5× as far as the rational agent at `k=0`, with intervals
excluding it. This is a calibration statement rather than gullibility: the cue is
80% reliable and the models behave as though it were more. Because the design
renders both wordings for every episode, over-movement costs variance rather than
bias.

They then diverge sharply. GPT under-shoots from `k=2` on (0.027 against the
oracle's 0.075) — it stops reading the background while a rational agent still
profits from it. Claude tracks the benchmark down and converges by `k=4`.
DeepSeek stays far above it at every rung.

### Direction is recovered, and stays recovered where it is measurable

| Forecaster | k=0 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|---:|
| Claude Opus 4.8 | 1.000 | 0.958 | 0.896 | 0.719 |
| DeepSeek V4 Pro | 1.000 | 0.854 | 0.833 | 0.792 |
| GPT-5.4 | 1.000 | 0.865 | 0.573 * | 0.573 * |
| Bayes oracle | 1.000 | 1.000 | 1.000 | 1.000 * |

All three read the background the right way round on **every** episode at `k=0`,
without being told that the wording was informative or which direction it
pointed. That is the cleanest positive evidence in the pilot that the mechanism
was inferred from the two worked examples rather than supplied.

`*` marks rungs where the cue moves the forecast less than 0.05, so direction is
unresolved rather than lost — the comparison degenerates into a coin flip on
rounding. The oracle's own figure falls to 0.667 by `k=8` for exactly this
reason, which is why the decay must not be read as a failure.

**Selectivity** — how far adding a mechanism-relevant background moves the
forecast, over how far adding equally detailed irrelevant prose moves it. Both
terms are **step-matched**: each is measured against the same episode's
no-context prompt, so `1.0` means irrelevant prose moves the forecast exactly as
much as relevant prose does.

| Forecaster | k=0 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|---:|
| Claude Opus 4.8 | 2.2 | 10.1 | 25.5 | 138.0 |
| DeepSeek V4 Pro | 1.4 | 1.8 | 1.4 | 1.7 |
| GPT-5.4 | 2.6 | 2.4 | — | — |
| Bayes oracle | ∞ | ∞ | ∞ | — |

Claude becomes sharply selective once City C has data: its irrelevant movement
falls from 0.171 to 0.001 while relevant movement stays measurable. GPT drops
*both* to near zero by `k=2` (0.027 and 0.001), so its ratio becomes undefined —
it stops reading the background at all rather than reading it selectively.
DeepSeek's irrelevant movement declines only from 0.280 to 0.084 and its ratio
sits at 1.4–2.0 at every rung with non-overlapping intervals against Claude: it
moves nearly as much for irrelevant prose as for relevant prose, at every level
of evidence. At `k=4` its irrelevant movement (0.143) is about half its relevant
movement (0.292).

The ratio is suppressed where relevant movement itself falls below 0.05, since a
ratio of two near-zero movements says nothing.

## Withdrawn readouts

Three framings were tried and removed as unsound rather than merely unclear. They
are recorded so the retraction is auditable and so they are not reintroduced.

**Aligned benefit / misleading cost.** The paired change in response error when
the background happens to be right, and when it happens to be wrong. For any
forecaster that shifts by `d` on the high-transmission wording and `-d` on the
buffered wording, the aligned change is `-d` and the misleading change is `+d`
**whatever `d` is**. The reassuring negative-then-positive shape was therefore
arithmetically forced by using the cue at all, not evidence of using it well, and
the pair carried exactly one number — the movement magnitude now reported
directly. Confirmed empirically: at `k=0`, half the cost-minus-benefit spread
reproduced the movement magnitude to within 0.01 for every model (Claude 0.375
against 0.378; DeepSeek 0.377 against 0.381; GPT 0.337 against 0.338).

**Excess susceptibility.** The model's misleading cost minus the oracle's, read as
"more movable than a rational agent". The oracle's no-context posterior reaches
0.9968 by `k=8`, so an 80%-reliable cue cannot shift it and its penalty collapses
from 0.225 to 0.003. The difference then simply restates the model's own penalty
while still carrying a label that implies the benchmark had comparable
opportunity to err. It did not. Every panel now plots levels side by side, so a
vanishing benchmark is visible as a line at zero.

**Net value of the background.** Mean response error across both cue conditions
against no context. On a design that renders both wordings for every episode a
symmetric shift cancels to first order, and it measured between 0.00 and 0.09 for
every forecaster including the oracle. Rejected before it reached the figure.

**A metric correction, not a withdrawal.** The selectivity ratio previously
divided the `cue_high` − `cue_low` *swing* by the `orthogonal` − `none` movement.
The numerator spanned two conditions and the denominator one, so it carried
roughly twice the leverage and inflated every ratio about two-fold (Claude read
4.4 at `k=0` rather than 2.2). Cross-model comparisons were unaffected because
all forecasters used the same definition, but the absolute values were not
interpretable and `1.0` did not mean parity. Both terms are now one-step changes
from the same baseline.

Because all three models see byte-identical prompts, DeepSeek's behavior on both
measures is a property of that model, not a design artifact.

## Where the forecasts sit

Mean absolute distance between each model's implied response and the two
prompt-visible anchors, no context:

| Model | k | Distance to frequentist | Distance to pooled references |
|---|---:|---:|---:|
| Claude Opus 4.8 | 2 | 0.008 | 0.388 |
| Claude Opus 4.8 | 8 | 0.008 | 0.364 |
| DeepSeek V4 Pro | 2 | 0.043 | 0.382 |
| DeepSeek V4 Pro | 8 | 0.062 | 0.354 |
| GPT-5.4 | 2 | 0.003 | 0.394 |
| GPT-5.4 | 8 | 0.011 | 0.365 |

This is the headline finding, and it is tight at 48 episodes. Once City C has two
or more cases, all three models sit essentially **on** the target sample mean and
far from reference-only pooling. They use the demonstrated cities where City C is
silent — at `k=0` their implied response is 0.605–0.641, the midpoint of the two
demonstrations — and to read the background cue. They then stop using them.

The cost is the green headroom in panel A, which they never enter. At `k=8` the
models sit at 0.794–0.885 MAE against 0.860 for the frequentist and 0.011 for the
ceiling: the entire remaining gap to the ceiling is left on the table.

**A correction from the eight-episode round.** That round showed DeepSeek at 0.300
and GPT at 0.413 MAE at `k=8` against 0.573 for the frequentist, which looked like
shrinkage toward the demonstrated cities. It was flagged there as resting on one
or two episodes. At 48 episodes it disappears entirely — all three models land
within 0.07 of the frequentist. The caution was correct and the apparent effect
should not be cited.

## Caveats

- The reference forecasters in `pilot_v2_summary.md` are computed on these 48
  episodes, so they differ slightly from the 120-episode design values in
  `validation_c2_v2.json` (frequentist at `k=8`: 0.860 here against 0.909).
- City A and City B still show six cases each, so at `k=8` City C's own record is
  richer than either demonstration. That is the intended upper anchor.
- Forecasts land outside the demonstrated response band in 42–48% of no-context
  cases at `k=8`. The band is never stated as a bound, so this is descriptive of
  exemplar anchoring, not a correctness criterion.
- Scoring restricts to prompts every model answered. Nothing was dropped in this
  round; the guard exists so an interrupted run cannot silently produce an
  unequal-sample comparison.

## Recommendation

Scale the frozen v2 task at `TARGET_CASES = 8` on episodes 48–119. No prompt
revision is required. Before the confirmatory run, complete the outstanding gate
in the plan — independent human review of the background templates — and fix the
episode count in advance. Raise the ladder toward 64 cases when the convergence
anchor itself becomes the claim.

## Artifacts

- `biased_news/data/three_city_c2_v2/pilot_v2/pilot_v2_diagnostics.png`
- `biased_news/data/three_city_c2_v2/pilot_v2/pilot_v2_summary.md`
- `biased_news/data/three_city_c2_v2/pilot_v2/pilot_v2_rows.jsonl`
- `biased_news/data/three_city_c2_v2/pilot_v2/responses_*.jsonl`
- `biased_news/data/three_city_c2_v2/design_diagnostics.png`
- `biased_news/eval/run_three_city_c2_v2_pilot.py`
- `biased_news/analysis/analyze_three_city_c2_v2_pilot.py`
