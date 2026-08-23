# Three-city C2 v7 confirmatory specification

This specification is frozen before any v7 model response is collected. Versions
v1–v6 are development pilots and are not confirmatory evidence for v7.

## Claim

The experiment tests whether a language forecaster recovers recurring response
mechanisms from two reference cities and applies that structure to a third city.
The structure-blind arm measures spontaneous use. The relevance-hint arm
measures whether the capability is elicitable with a minimal cue.

## Design

- 120 fresh episodes, seeds 70000–70119.
- Hidden responses 0.90 and 0.30, balanced exactly.
- The high-response A/B label and three context paraphrase families are exactly
  balanced.
- Six representative independent cases are displayed for each reference city.
- City C prefixes are \(k \in \{0,1,2,4\}\).
- Target backgrounds are truthful mechanism-relevant, absent, or
  length-matched orthogonal prose.
- Every case starts at 50 and receives net news +8. Poll measurements vary
  across otherwise independent cases.

## Prompt manipulation

Blind and hint prompts are byte-identical except that the hint arm inserts:

> When forecasting City C, consider whether the earlier cities' response
> patterns and background descriptions are informative.

Both arms receive the same statement about case-to-case variation, reasoning
instruction, token budget, response schema, and one-sentence rationale limit.

## Confirmatory outcomes

The primary condition is the truthful mechanism-relevant background at \(k=2\).
There are exactly two primary outcomes:

1. Paired hint improvement in forecast MAE:
   \(e_{\text{blind}}-e_{\text{hint}}\), so positive values favor the hint.
2. Paired hint improvement in the structure-use coefficient \(\beta\), estimated
   from
   \[
   \hat y_{\text{model}}-\hat y_{\text{target-only}}
   = \alpha + \beta
   (\hat y_{\text{hierarchical}}-\hat y_{\text{target-only}})+\epsilon.
   \]
   \(\beta=0\) is target-only behavior and \(\beta=1\) matches the public-data
   hierarchical reference.

Blind and hint forecasts are paired on the same episode and numeric prompt.
Uncertainty is a deterministic 10,000-draw percentile bootstrap resampling
episodes as clusters. Each frontier model is reported as a separate replication;
models are not treated as independent samples from a population of models.

## References

- Naive: population midpoint, ignoring city-specific evidence.
- Target-only: City C's average, with the midpoint at \(k=0\).
- Pooled references: average A and B while ignoring their distinct patterns.
- Empirical hierarchical: estimate both prototypes and residual scale only from
  displayed A/B records, match the three displayed backgrounds with a frozen
  analyst-coded semantic codebook, update using displayed City C cases, and
  treat a truthful relevant background as one explicitly declared pseudo-case
  at the semantically matching reference response. It does not consult City
  C's hidden type or the evaluator's target-match label.
- DGP oracle: privileged ceiling that knows City C's hidden response. It is
  labeled as privileged and is not a same-information comparator.

The one-pseudo-case convention is a transparent diagnostic reference, not a
claim of uniquely optimal context weighting. Sensitivity to zero, one-half, and
two pseudo-cases is secondary.

## Secondary analyses

- MAE and \(\beta\) across \(k=0,1,2,4\).
- Relevant versus absent and orthogonal backgrounds.
- Completion and parsing rates.
- Sensitivity of the empirical hierarchical reference to context weight.
- A single-call, temperature-zero analysis is primary. Any repeated-completion
  robustness run is labeled secondary and is not averaged into the primary.

No pattern-classification accuracy, post-hoc “right side” metric, or unregistered
subgroup is confirmatory.

An obtained but unparseable completion is a terminal parsing failure and is not
selectively rerun. A transport failure before any completion is received may be
retried, and response and parsing rates are reported separately.

For each model, the preregistered elicitation-effect criterion requires both
hint-effect point estimates to be positive and both 95% bootstrap intervals to
exclude zero in the favorable direction. Blind and hint beta levels and all
model-specific exceptions are reported without suppressing discordant evidence.
Blind beta is interpreted directly with its interval; failure to reject zero is
not treated as proof of no spontaneous structure use. No claim about a
population of models is made from three named systems.
