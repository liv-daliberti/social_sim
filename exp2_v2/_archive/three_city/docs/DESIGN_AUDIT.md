# Experiment 2 v2 design audit

## Bottom line

The first three-city pilot is **not sufficient evidence for the paper's full C2
criterion**. It is a useful pilot of narrow latent-class recovery, but the prompt
hands the forecaster almost the entire hypothesis space:

- exactly two response types exist;
- City C matches A or B with prior probability 0.5;
- the state persistence is 0.9;
- process and poll noise are approximately 1 and 2 points;
- the hidden quantity is a fixed points-per-news gain;
- the response must explicitly report both the type probability and gain.

A successful response can therefore come from following a supplied statistical
model and performing arithmetic. It need not use background knowledge to
propose or interpret an underlying mechanism. The task addresses the "recover a
latent state" portion of C2, but not the paper's stronger requirement:

> use background knowledge to infer the underlying mechanisms, constraints, and
> contextual factors that determine how the event is likely to unfold.

The pilot results should be treated as design evidence, not confirmatory model
results.

## What the stopped pilot actually measured

The pilot asked whether a model could:

1. estimate two scalar response gains from two reference histories;
2. classify City C as sharing A's or B's exact scalar gain;
3. update that classification as City C observations accumulated; and
4. apply the inferred gain to four counterfactual news shocks.

Those are worthwhile behaviors. In particular, comparing the stated type
posterior with the gain implied by the forecasts measures whether a latent
belief propagates into predictions. But the latent "structure" is only a scalar
coefficient in a fully disclosed equation.

The cross-city component is also not automatically C3. All three cities share
the same variables, prompt language, and domain. It is cross-instance induction
within one environment, not transfer to a held-out environment with different
surface details.

## Why GPT and DeepSeek looked strange

The frontier jobs were intentionally stopped, leaving unequal and small samples:

| Model | Completed triplets |
|-------|-------------------:|
| Claude Opus 4.8 | 10 |
| DeepSeek V4 Pro | 19 |
| GPT-5.4 | 42 |

They cannot support a fair leaderboard comparison. They do reveal prompt and
readout problems.

Across every prefix in the partial records:

| Model | Behavioral gains outside [0,1] | Mean absolute gap between stated and behavioral gain |
|-------|--------------------------------:|-----------------------------------------------------:|
| Claude Opus 4.8 | 12 / 60 (20.0%) | 0.003 |
| DeepSeek V4 Pro | 16 / 114 (14.0%) | 0.093 |
| GPT-5.4 | 55 / 252 (21.8%) | 0.005 |

Thus out-of-range gains are not uniquely a DeepSeek/GPT failure. Claude usually
made its stated gain and four forecasts mutually consistent, while DeepSeek did
so less reliably. More importantly, all three often produced forecasts whose
implied gain did not agree with the gain implied by their reported type
probability. The mean absolute belief-to-forecast gain gap over prefixes was
roughly:

- Claude: 0.08–0.13;
- GPT: 0.13–0.19;
- DeepSeek: 0.13–0.35.

For the true low-gain cities, average behavioral estimates remained well above
the true 0.25 even after five polls. For true high-gain cities, later DeepSeek
and GPT estimates often exceeded 1.0. This pattern is consistent with models
doing ad hoc arithmetic on noisy cumulative poll changes, rather than cleanly
filtering the supplied mean-reverting state equation.

The four-counterfactual elicitation compounds the problem. It simultaneously
tests:

- latent classification;
- state filtering;
- gain estimation;
- mean-reversion arithmetic;
- four-way numerical consistency; and
- strict JSON instruction following.

Failure cannot be attributed cleanly to latent-structure recovery.

## Relationship to Figure 1

Figure 1 works because the same sparse evidence—two heads—supports different
forecasts under different inferred processes:

- naive: assume a fair coin and ignore the setting;
- frequentist: use only the two observed flips;
- world-model hypothesis: use the "carnival game" context to place prior mass on
  a rigged coin, while retaining uncertainty after only two flips.

The distinctive step is **contextual prior formation**. The prompt does not tell
the forecaster the exact latent family or prior. The phrase "carnival game"
activates background knowledge about plausible mechanisms.

The current city pilot instead supplies the latent family, prior, dynamics, and
noise. Its 50/50 average is analogous to a Bayes calculation after the world
model has already been specified—not to the act of constructing that model from
context.

## Corrected construct: two linked subtests

Trying to force the entire C2 claim into one score creates an information-parity
problem. A fully specified synthetic DGP is fair and exactly scoreable, but
leaves little role for background knowledge. An underspecified contextual task
requires background knowledge, but an exact numerical oracle necessarily knows
more than the model.

The corrected experiment therefore separates two claims.

### C2a: structure-blind latent-mechanism recovery

Question: given two demonstrated mechanisms and sparse evidence from City C,
does the forecaster infer which mechanism applies and use it predictively?

- The numerical transition is deliberately simple, but its structure is **not
  described to the model**.
- The model is not told that there are two types, that City C matches A or B,
  that a gain exists, or that any prior is 50/50.
- Only the stable city response type is hidden.
- The target begins with weakly diagnostic evidence, then receives strongly
  diagnostic evidence.
- The omniscient Bayes oracle knows the hidden DGP and is explicitly labeled a
  ceiling, not an information-matched behavioral comparator.
- Observation-matched statistical baselines must estimate their structure from
  the same two reference tables and City C history.
- This subtest supports **latent recovery and coherent application**, but makes
  no standalone claim about background knowledge.

### C2b: contextual mechanism induction

Question: does mechanism-relevant natural-language context supply a useful,
selective prior, and can later numerical evidence override it?

- Reference cities pair distinct information-environment descriptions with
  distinct response patterns.
- City C receives a semantically related but lexically different context.
- Paired variants hold every number fixed while swapping relevant, orthogonal,
  or absent context.
- Early target evidence is intentionally ambiguous, so context can help.
- Later target evidence is diagnostic, so a misleading contextual cue should be
  overridden.

This is the city analogue of the carnival clue. It tests the background-
knowledge portion of C2 behaviorally through selective context use and evidence
override. It is analyzed through paired contrasts rather than by pretending an
LLM was handed the oracle's exact contextual prior.

The full C2 claim requires success on both subtests:

```text
recover a hidden mechanism from controlled evidence
                         +
use causal context selectively, then override it when evidence demands
                         =
defensible behavioral evidence for C2
```

## Corrected world

To isolate the construct, remove the hidden opinion/Kalman-filter layer:

```text
poll_0 = 50
poll_t = poll_(t-1) + g_z * news_t + epsilon_t
epsilon_t ~ Normal(0, 2)
z in {open-information, buffered-information}
g_open = 1.00
g_buffered = 0.25
```

The current poll is the observed state. The only hidden variable is the stable
mechanism/type. This avoids attributing failures in state filtering or
mean-reversion arithmetic to C2.

Reference cities receive several large, balanced news shocks, making their two
response patterns clear. City C receives:

1. no local evidence (`k=0`);
2. a small, weakly diagnostic shock (`k=1`);
3. a large, diagnostic shock (`k=2`);
4. an opposite-sign confirmation shock (`k=3`).

The equation above exists only in the evaluator and preregistration. It is never
included in the model prompt.

At each prefix the forecaster makes one next-poll prediction under a single
held-out shock. This mirrors Figure 1's one forecast and yields a behavioral
implied gain:

```text
g_behavior = (predicted_next_poll - current_poll) / next_news
```

The primary prompt does not request a type probability or gain. Type recovery is
inferred behaviorally from the forecast's proximity to the two hidden response
regimes. An optional mechanism probe, if used, must be a separate secondary
elicitation and cannot define success.

## Context conditions

Every underlying numerical episode is rendered as paired prompt variants:

| Condition | Purpose |
|-----------|---------|
| none | Numerical evidence only. |
| relevant-aligned | Mechanism-relevant context points toward the true type. |
| relevant-misleading | The same kind of noisy cue points toward the wrong type. |
| orthogonal | Equally detailed context has no causal connection to response. |
| context-swap pair | Same numbers, high-response versus buffered cue. |

The relevant descriptions concern how national campaign news reaches residents,
not polling-company ownership or partisan valence. Multiple paraphrase families
must be used, with low lexical overlap between reference and target contexts.

The prompt does **not** say that the contextual descriptions are useful, causal,
or reliable. Discovering that relationship from the two examples is part of the
task. The preregistered contextual oracle privately assigns the relevant cue a
reliability of 0.8; model performance is not interpreted as failure merely for
differing from that exact number. The primary contextual tests are directional
and paired.

## Forecasters

1. **Naive:** predicts no news response (the current poll again) at every prefix.
2. **Frequentist:** unregularized through-origin regression using only City C's
   observed changes and news.
3. **Pooled exemplar regression:** estimates one response coefficient by pooling
   the two reference-city records, with no latent classes.
4. **Two-prototype empirical baseline:** estimates separate response patterns
   from A and B, then decides from City C's data how to weight them. It is not
   handed the true gains or noise level.
5. **Target-only Bayes diagnostic:** knows the evaluator's two gains and 50/50
   prior, but ignores references and context; labeled as structure-informed.
6. **Context-aware Bayes oracle:** additionally uses the preregistered contextual
   reliability; it is an omniscient ceiling, not an information-matched baseline.
7. **Clairvoyant ceiling:** sees the hidden type; reported only as a scale
   reference.
8. **Language model:** sees only the rendered examples, context, history, and
   forecast question.

The context-aware oracle is an upper reference, not evidence that the LLM was
given an identical numerical prior.

## Preregistered primary outcomes

### C2a

- next-poll MAE by prefix;
- behavioral gain error by prefix;
- behavioral type accuracy, defined by which hidden response regime the implied
  gain is closer to.

### C2b

- aligned-context improvement at `k=0` and `k=1` over the no-context prompt;
- near-zero response to orthogonal context;
- paired forecast movement under the context swap;
- evidence override: reduction of a misleading context effect after the
  diagnostic `k=2` observation;
- robustness across context paraphrase families.

The type probability is never accepted as proof by itself. It must propagate to
the forecast.

## Reproducibility and fairness gates before any new API run

- Freeze task JSONL before model evaluation.
- Assert automatically that no rendered prompt contains the hidden type names,
  the words "gain" or "two types," the DGP equation, numeric noise levels, or a
  50/50 prior.
- Use balanced true types, reference-label swaps, shock signs, and context
  conditions.
- Fix and publish the seed list.
- Hash every prompt template and task manifest.
- Use temperature 0 where supported and record all deployment parameters.
- Include a repeated-call reliability subset.
- Validate label-swap invariance for every deterministic baseline.
- Validate that `k=1` is weakly informative and `k=2` is strongly informative
  using oracle likelihood ratios.
- Human-check context templates for mechanism relevance, valence, length, and
  lexical-overlap balance.
- Do not compare models until all share the same completed episode set.
- Never treat partial unequal-n runs as final evidence.

## Claim language

If only C2a succeeds:

> Agents can infer and apply a latent response pattern from two examples and
> sparse controlled evidence without being told the underlying structure.

If C2a and the selective-context/evidence-override tests in C2b succeed:

> Agents use mechanism-relevant contextual knowledge to form priors over hidden
> response regimes, update those priors with sparse evidence, and propagate the
> inferred mechanism into forecasts.

The stronger sentence is the one that matches the paper's C2 definition.
