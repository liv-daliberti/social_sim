# Three-city C2 v8 confirmatory specification

This replacement experiment is frozen before any v8 model response is
collected. Versions v1–v7 are development studies and are not pooled with v8.
In particular, the partial v7 collections were stopped after design review
showed that an always-aligned City C background made the hidden target pattern
nearly deterministic at \(k=0\).

## Claim and Figure-1 mapping

The experiment tests the two components of the paper's C2 criterion without
collapsing them:

1. **C2a, latent-structure recovery:** infer two recurring response patterns
   from Cities A and B, then combine that structure with sparse City C evidence.
2. **C2b, contextual mechanism induction:** respond selectively to a
   mechanism-relevant target description and override a misleading description
   as City C evidence accumulates.

The main curve mirrors Figure 1. A City-C-only regression is the analogue of
using the observed flips alone. The A/B/C regression-mixture is the
world-model analogue: it estimates the two demonstrated patterns, begins City C
at equal weight on them, and updates those weights using City C's cases. At
\(k=0\), neither City C's type nor a target-type cue is available, so both
methods reduce to the displayed A/B midpoint and expected forecast MAE is high,
not zero.

## Frozen world and sample

- 120 fresh episodes, seeds 80000–80119.
- Every row is an independent polling case beginning at 50.0 with net news
  \(+8\).
- The two hidden response patterns are 1.00 and 0.25 poll points per news point.
- Case noise has evaluator-only standard deviation 3.25.
- Each reference city shows six representative cases. Reference residuals are
  centered to mean zero while retaining their dispersion, as in the audited v2
  design.
- City C prefixes are \(k\in\{0,1,2,3,4,6,8\}\).
- Target type, which A/B label is the high-response city, and three context
  paraphrase families are exactly balanced.

The model is never shown the equation, response values, noise scale, prior
probability, hidden type, evaluator match label, or context reliability.

## Three prompt arms

All numerical content, city descriptions, sampling language, forecast question,
reasoning instruction, response schema, temperature, and token budget are
identical across arms.

The **structure-blind** arm adds no structural instruction.

The **relevance-hint** arm adds exactly:

> When forecasting City C, consider whether the earlier cities' response
> patterns and background descriptions are informative.

The **strong relevance-hint** arm instead adds exactly:

> Cities in this task follow two recurring response patterns, demonstrated by
> Cities A and B. City C follows one of those patterns. Use City C's completed
> cases and any relevant background information to decide how much weight to
> place on each pattern when forecasting.

The strong arm is a preregistered positive control. It discloses the two-pattern
hypothesis but not City C's pattern, the numerical responses, a prior,
likelihood, equation, noise scale, or answer.

## Main C2a structure curve

The main curve uses the **no-target-background** condition only. Cities A and B
retain their mechanism descriptions and cases; City C has no type cue.

The two prompt-visible comparisons are:

- **Use City C only:** through-origin response regression on completed City C
  cases. At \(k=0\), where it is unavailable, the displayed A/B midpoint is the
  explicit fallback.
- **Combine A/B/C:** fit one response regression to A and one to B, estimate a
  residual scale from those displayed records, begin City C at 50/50 over the
  two fitted patterns, and update the weights by the likelihood of City C's
  displayed cases.

The preregistered sparse anchor is \(k=2\), matching Figure 1's two
observations. The \(k=8\) endpoint tests convergence. For each named model and
arm we report:

- forecast MAE;
- paired improvement over City C only;
- the cross-city structure-use coefficient
  \[
  \hat y_{\mathrm{model}}-\hat y_{\mathrm{C-only}}
  =\alpha+\beta
  (\hat y_{\mathrm{A/B/C}}-\hat y_{\mathrm{C-only}})+\epsilon,
  \]
  where \(\beta=0\) follows City C only and \(\beta=1\) matches the A/B/C
  regression-mixture;
- relevance-hint minus blind and strong-hint minus blind contrasts.

The full \(k\) curve is reported, with \(k=2\) and \(k=8\) visually marked.

## Separate C2b context panel

Each numeric episode, arm, and prefix is rendered with four target-background
conditions:

- none;
- length-matched orthogonal prose;
- a high-transmission mechanism description;
- a buffered/low-transmission mechanism description.

The high and low descriptions are rendered for **every** hidden target type.
They are therefore crossed with truth rather than always aligned. No aligned
condition is selected as the main MAE curve.

For each model and prompt arm, paired within episode, we report:

1. Relevant movement
   \[
   R_k=\tfrac12\left(
   |\hat y_{\mathrm{high}}-\hat y_{\mathrm{none}}|
   +|\hat y_{\mathrm{low}}-\hat y_{\mathrm{none}}|
   \right).
   \]
2. Orthogonal movement
   \(O_k=|\hat y_{\mathrm{orthogonal}}-\hat y_{\mathrm{none}}|\).
3. Cue direction
   \(\Pr(\hat y_{\mathrm{high}}>\hat y_{\mathrm{low}})\).
4. Misleading-cue displacement: the absolute difference between the
   truth-opposed cue forecast and the same episode's no-context forecast.
5. Override from \(k=0\) to \(k=8\): the paired reduction in
   misleading-cue displacement.

The intended C2b signature is measurable, correctly directed early movement;
substantially smaller orthogonal movement; and reduced misleading-cue
displacement by \(k=8\). Context effects are also plotted beside the privileged
v2 rational benchmark holding 0.80 cue reliability, never subtracted from it or
described as information-matched.

## Inference, stopping, and reporting

- All uncertainty intervals are deterministic 10,000-draw percentile
  bootstraps resampling episodes as clusters.
- The three named models are separate replications, not a random sample of
  models.
- Contrasts across arms, contexts, and prefixes use only episodes with parsed
  responses in every cell required by that contrast.
- One received completion is retained. An unparseable received completion is a
  terminal parsing failure and is not selectively rerun. A transport failure
  before a completion may be retried.
- Completion and parsing rates are reported by model and arm.
- The fixed run contains 3,360 calls per model-arm, 10,080 calls per model, and
  30,240 calls total.
- Main and context figures show all three models on the same facet and use
  separate facets for structure-blind, relevance hint, and strong relevance
  hint.
