# What the three-city pilot found

48 episodes, 1,344 prompts per model, identical prompts for Claude Opus 4.8,
DeepSeek V4 Pro and GPT-5.4. Design check, not a confirmatory result.

Figure: `biased_news/data/three_city_c2_v2/pilot_v2/pilot_v2_findings.png`

**Reading panel 3.** Everything is re-expressed as signed distance from the
decision boundary, so the horizontal axis line *is* the boundary: above it means
"called this a responsive city", below means "called it buffered". Bars are means,
dots are individual cities, dashed lines are the two true values at ±0.375. A bar
reaching its dashed line has committed fully to the value its own classification
implies.

## The one-sentence version

**All three models correctly work out which kind of city they are looking at, and
then decline to act on it — they forecast the target city's noisy average instead
of the value their own conclusion implies.**

## The setup, in plain terms

The model sees two worked cities. City A responds fully to news (a +8 story moves
its poll about 8 points). City B is buffered (about 2 points). It is never told
that two kinds exist, that a coefficient exists, or that City C is one of the two.
It then sees between 0 and 8 completed City C cases and forecasts one more.

The interesting question is how it splits its answer between two sources: the two
worked examples, and City C's own noisy record.

## How "which kind of city" is read off

Nothing in the prompt asks, and nothing in the reply states it. It is decoded from
the single number returned:

```
implied response = (predicted_poll - 50.0) / net_news
midpoint of the two demonstrated values = the decision boundary
```

Above the boundary is scored "responsive", below it "buffered". Asking directly
would leak that City C *is* one of the two, which is the structure being withheld;
v1 asked for coupled self-reports and was retired partly for that reason.

## Finding 1 — with no target data, they use the two examples

At `k=0` there are no City C cases, so anything other than 50.0 has to come from
Cities A and B. All three land on the midpoint of the two demonstrations (0.625):

| Model | mean implied response at `k=0` | sd |
|---|---:|---:|
| Claude Opus 4.8 | 0.619 | 0.036 |
| DeepSeek V4 Pro | 0.612 | 0.039 |
| GPT-5.4 | 0.605 | 0.048 |

Forecast error there is 2.95–3.11 poll points against 3.000 for a forecaster that
knows the entire hidden generative process. **With nothing else to go on they are
already optimal**, and the rationales say so unprompted — *"Without City C
background, I average the two cities' mean responses to +8 news"* (Claude);
*"resembling City A's strong pass-through of favorable news"* (Claude, `k=2`).

This is the pilot's evidence that the structure was extracted from two examples.

## Finding 2 — the forecast lands on the correct side, but that proves less than it looks

Fraction of cities put on the correct side, with no background text:

| Forecaster | k=0 | k=2 | k=8 |
|---|---:|---:|---:|
| Claude Opus 4.8 | 0.542 | 0.917 | 1.000 |
| DeepSeek V4 Pro | 0.438 | 0.875 | 0.958 |
| GPT-5.4 | 0.490 | 0.917 | 1.000 |
| Best possible | 0.500 | 0.917 | 1.000 |
| **City C's average alone — provably never reads A or B** | — | **0.917** | **1.000** |

`k=0` is chance by construction. From two cases onward the models match the
ceiling exactly — but so does the plain average of City C's own cases, which by
construction never looks at the reference cities.

**So this curve is not evidence of inference.** It measures whether the forecast
falls on the correct side of the boundary, and averaging City C achieves that for
free because its sample mean usually lands there. An earlier draft of this
document titled the corresponding panel "They DO infer the two kinds of city";
that was an overclaim, and the panel now plots the averaging baseline on top of
the curve so the confound is visible rather than hidden.

What the curve does establish is the premise for Finding 3: by `k=8` the models
have the side right on essentially every city.

## Finding 3 — the forecast does not follow the identification

Forecast error in poll points, no background:

| Forecaster | k=0 | k=2 | k=8 |
|---|---:|---:|---:|
| Claude Opus 4.8 | 2.946 | 1.707 | 0.885 |
| DeepSeek V4 Pro | 3.108 | 1.818 | 0.794 |
| GPT-5.4 | 2.994 | 1.759 | 0.867 |
| Just average City C's own cases | — | 1.755 | 0.860 |
| Use both examples and City C | 3.000 | 0.774 | 0.011 |

From `k=2` onward every model sits on the "just average City C" line. In implied
response units their distance from that plain average is 0.003–0.062, while their
distance from the examples is 0.354–0.459. The examples are dropped the moment
City C has numbers of its own.

**The sharpest way to say it.** At `k=8` Claude and GPT-5.4 classify City C
correctly on **100%** of episodes. A forecaster acting on that same correct
classification would predict the prototype value and carry 0.011 poll points of
error. They carry 0.885 and 0.867. DeepSeek classifies correctly on 95.8% and
carries 0.620.

So this is not a knowledge failure. They know the answer is "the responsive kind",
which implies 58.0, and they predict 57.3 — City C's sample mean. Knowing which
of two values is right should collapse the forecast onto it. It does not.

DeepSeek's two wrong-side cases at `k=8` are both buffered cities placed on the
responsive side; they are visible as red crosses in panel 3, and they are the only
wrong-side cases any forecaster has at that rung.

### It is a variance failure, not hedging

The obvious competing explanation is that the models shrink toward the middle —
that knowing City C is "probably responsive" produces a cautious answer part-way
between the two values. Panel 3 rules that out. Measured as distance from the
decision boundary at `k=8`, where the ideal is ±0.375:

| Forecaster | responsive cities | buffered cities |
|---|---|---|
| Best possible | +0.375, sd 0.001 | −0.372, sd 0.011 |
| Claude Opus 4.8 | +0.342, sd 0.122 | −0.386, sd 0.153 |
| GPT-5.4 | +0.352, sd 0.127 | −0.378, sd 0.138 |
| DeepSeek V4 Pro | +0.351, sd 0.106 | −0.319, sd 0.193 |

The **means** are 85–103% of ideal. Nobody is hedging: on average they commit to
very nearly the right value. The **spreads** are 10–190× the ceiling's.

So the entire gap in the second table is variance. Each individual forecast
tracks that episode's noisy sample mean rather than snapping to the prototype, and
the errors cancel in the average while never cancelling in any single forecast.
That is a sharper claim than "they don't use the structure": they use it to decide
*which* value, and then decline to use it to decide *the* value.

## Why this is the interesting result for C2

C2 asks whether an agent can use background knowledge to infer the mechanisms and
constraints that govern how an event unfolds. This pilot separates that into two
steps and finds they come apart:

- **recovering** the latent structure — succeeds, at the ceiling;
- **propagating** it into a quantitative forecast — fails, leaving roughly 0.8 of
  0.9 available poll points unused.

A benchmark that scored only classification would call this a pass. A benchmark
that scored only forecast error would call it a failure and not say why. The
design gets both because it asks for one number and reads the implied structure
out of it.

## Secondary result — background text

Each episode is also rendered with four City C background descriptions: none, an
irrelevant one, and two mechanism-relevant ones pointing opposite ways, with every
number held fixed.

- All three read the background **the right way round on every episode** at `k=0`.
  Nothing told them the wording mattered or which way it pointed.
- All three weight it about 1.5× as heavily as its true 80% reliability warrants.
- They then diverge: Claude converges to the rational weighting by `k=4`; GPT
  stops using the background at all from `k=2`; DeepSeek keeps over-weighting it
  and is also the only model moved by the *irrelevant* description (0.084–0.280
  against 0.001 for the others).

Details, confidence intervals, and three readouts that were tried and withdrawn as
unsound are in [PILOT_V2_AUDIT.md](PILOT_V2_AUDIT.md). The six-panel
`pilot_v2_diagnostics.png` is the supplementary view organised by metric.

## What this does not establish

- 48 of the 120 frozen episodes. Confirmatory runs go on episodes 48–119 and must
  not pool these.
- The interim ladder stops at 8 City C cases, where "just average City C" is still
  0.85 poll points short of the ceiling. Whether the gap would finally close in
  the models' favour at 64 cases is untested.
- Model rankings here are directional. The design check, not the comparison, is
  what the pilot was for.
