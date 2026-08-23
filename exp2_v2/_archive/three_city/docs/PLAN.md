# Experiment 2 v2: infer from two cities, forecast a third

> **Pilot status (2026-07-27):** A four-episode, 48-prompt-per-model diagnostic
> completed for Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4. It confirmed that
> models infer and use the unstated response patterns, but exposed a
> baseline-versus-current-poll ambiguity in the reference histories. Do not
> scale the frozen v1 task before the targeted revision in
> [PILOT_AUDIT.md](PILOT_AUDIT.md).
>
> **Revision status:** The targeted independent-case redesign is frozen as
> `three_city_c2_v2`. See [C2_V2_DESIGN.md](C2_V2_DESIGN.md). It removes all
> temporal dynamics, uses the same moderate news value in every repeated case,
> and extends City C to three observations.
>
> **v2 pilot status (2026-07-27):** A 48-episode, 1,344-prompt-per-model run
> completed for all three models with zero parse failures, on a City C ladder
> reaching eight cases. Both v1 defects are fixed and the design is cleared to
> scale; see [PILOT_V2_AUDIT.md](PILOT_V2_AUDIT.md). The canonical experiment is
> `three_city_c2_v2`. Episodes 0–47 are spent; run the confirmatory sample on
> episodes 48–119 and do not pool the pilot into it.
>
> **Start here for results:** [FINDINGS.md](FINDINGS.md) states what the pilot
> found in plain terms. [PILOT_V2_AUDIT.md](PILOT_V2_AUDIT.md) carries the method
> detail, confidence intervals, and the readouts that were withdrawn.

## Canonical design

The canonical experiment is `three_city_c2_v2`. It asks whether a forecaster
can discover a useful response pattern from two worked city examples, use
background descriptions selectively, update from City C's own observations, and
make one held-out forecast.

The model is **not told the structure**. In particular, no prompt says:

- how many city types or response patterns exist;
- that City C matches either reference city;
- that a response coefficient exists;
- the data-generating equation or noise level;
- a prior probability or contextual-cue reliability; or
- which background details are useful.

Those details exist only in the private evaluator. The model sees City A and
City B's background descriptions and case tables, City C's available background
and completed cases, and one announced net-news value for a new case. It returns
one end-of-case poll forecast.

The complete rationale and claim boundary are in
[DESIGN_AUDIT.md](DESIGN_AUDIT.md), and the minimal-design rationale is in
[C2_V2_DESIGN.md](C2_V2_DESIGN.md). The frozen configuration is
`biased_news/configs/three_city_c2_v2.yml`.

## Why this matches Figure 1

Figure 1's two coin flips do not announce a coin model. The carnival setting
helps a forecaster propose a plausible mechanism, and later observations should
still be able to overturn that expectation.

The city experiment uses the same logic:

```text
two city examples + their background
                    |
                    v
infer a possible response pattern for City C
                    |
          weak City C evidence
                    |
                    v
make one next-poll forecast
                    |
       diagnostic City C evidence
                    |
                    v
retain or override the initial interpretation
```

This supports two separate tests:

1. **C2a — structure-blind recovery:** can the forecaster recover and apply the
   numerical pattern without being told what to look for?
2. **C2b — contextual induction:** does a mechanism-relevant description affect
   sparse forecasts, does irrelevant detail not affect them, and does strong
   numerical evidence override a misleading description?

## Paired task design

City A and City B each show six completed cases. Each numerical episode is then
rendered at every City C prefix from 0 to 8 completed cases, so the target's own
evidence accumulates one independent measurement at a time. Curves and design
gates report the ladder `0, 1, 2, 3, 4, 6, 8`: adjacent full-ladder rungs differ
by less than the sampling error at 120 episodes.

Eight is an interim upper anchor. The ceiling holds a two-point prior and so
identifies the city outright, while the frequentist estimates a continuous mean
and only improves as `1/sqrt(k)`; the gap therefore narrows along the ladder
without closing, and reaching a ~0.3-point gap needs about 64 cases.

The same episode and prefix are rendered with four City C backgrounds:

- none;
- equally detailed but orthogonal;
- a high-response information environment;
- a buffered information environment.

The paired prompts hold every number and all reference material fixed. Only
City C's background changes. True responses, A/B labels, and three paraphrase
families are balanced.

## Comparison forecasters

Two information-matched anchors bracket the model from opposite directions, plus
one ceiling:

- **City C's own average only** (`frequentist` in the answer key): estimates City
  C's response from City C's completed cases and nothing else. Because every case
  carries the same news value, this collapses to the plain average of City C's
  observed end-of-case polls. Undefined at `k=0`. Figures use the plain-language
  name, since "frequentist" hides what it actually computes.
- **City A+B average only** (`pooled_exemplar`): the mirror image — one response
  estimated from the two demonstrated cities, never City C. Flat in `k`.
- **Target-only Bayes ceiling:** knows the private numerical DGP but ignores
  reference cities and context. Structure-informed, not information-matched.
- **Context-aware Bayes oracle:** additionally knows the evaluator's contextual
  reliability. Used as the *rational benchmark* for background contrasts: it fixes
  how far a forecaster holding the true 80% cue reliability should move, so
  over- and under-weighting are both visible. Its level is plotted next to each
  model's, never subtracted from it — the oracle's no-context posterior reaches
  0.997 by `k=8` and so becomes unmovable, which a difference would conceal.
- **LLMs:** receive only the structure-blind prompt.

The primary comparisons against an LLM are the two information-matched anchors.
The Bayes methods are ceilings, not claims that the model received equivalent
information.

Naive, two-prototype, and clairvoyant forecasters were removed: the first two
were uninformative and hand-built for this exact task, and the clairvoyant
prediction is the scoring target restated.

## Outcomes

The primary response is the one next-poll forecast. From it the evaluator
computes:

- absolute error against the noiseless expected poll;
- distance from the hidden response pattern;
- which demonstrated response pattern the forecast is behaviorally closer to;
- separation between the two hidden kinds of City C as `k` grows;
- distance from the target-only and pooled-reference anchors, which reveals how
  much weight the forecast places on the demonstrated cities;
- how far adding a mechanism-relevant background moves the implied response,
  against the same episode's no-context prompt, reported as a level beside the
  rational benchmark's own level rather than as a difference;
- whether the movement runs in the direction the wording implies; and
- selectivity — movement under a mechanism-relevant background over movement
  under equally detailed irrelevant prose, both step-matched against the same
  no-context prompt so 1.0 means parity.

Paired quantities carry 2.5–97.5 percentile bootstrap intervals over episodes.

No self-reported type probability or coefficient is needed for success.

## Reproducible workflow

From `exp2_v2/biased_news`:

```bash
# Inspect a prompt; this makes no API call.
python eval/build_three_city_c2_v2_tasks.py --dry-run

# Freeze the balanced public task file and private scoring key.
python eval/build_three_city_c2_v2_tasks.py --n 120

# Run all design and non-disclosure checks.
python eval/validate_three_city_c2_v2_tasks.py \
  data/three_city_c2_v2/tasks_c2_v2.jsonl \
  --answer-key data/three_city_c2_v2/answer_key_c2_v2.jsonl \
  --report data/three_city_c2_v2/validation_c2_v2.json

# Evaluate one model. The runner never opens the answer key.
MODEL=gpt-5.4 sbatch --export=ALL,MODEL=gpt-5.4 \
  eval/slurm_three_city_c2_v2_pilot.sh

# Join saved forecasts to hidden conditions and gold values.
python analysis/analyze_three_city_c2_v2_pilot.py
```

`--episodes/--prefixes/--variants` select a subset; the defaults are episodes
0–47 at the reported ladder, which the pilot has already spent. Pass
`--episodes {48..119}` for a confirmatory sample. Runs are resumable: completed
task IDs are skipped, with the stored prompt hash verified against the frozen
file before reuse.

No confirmatory evaluation should begin until the automated gates pass and an
independent human has reviewed the context templates for mechanism relevance,
valence, length, and lexical-overlap balance.

## Status of earlier work

`engine/three_city.py`, `eval/run_three_city.py`, and the stopped partial API
outputs are retained as a non-confirmatory pilot. That pilot explicitly
disclosed the latent family and requested several coupled quantities, so it
does not establish the paper's full C2 criterion and must not be combined with
the canonical results.

The v1 structure-blind task (`three_city_c2`, `pilot_v1`) is frozen as the audit
trail for [PILOT_AUDIT.md](PILOT_AUDIT.md). Its temporal ambiguity makes its
numerical forecasts non-comparable to v2, so it must not be pooled either.

`LEGACY_PLAN.md` is the plan copied from the original experiment folder.
