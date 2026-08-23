# Three-city C2 v8 full k=1...12 replacement specification

This amendment was frozen on 2026-08-04 before the replacement collection was
launched. It follows review of the separately preserved, early-stopped v8 n=50
collection. The old responses are not resumed or pooled: this is a clean
120-episode rerun of all three prompt arms on all three named models, written to
the separate `full_k1_12` run directory.

The changes requested for this replacement are:

1. reporting begins at \(k=1\), not \(k=0\);
2. every integer prefix is measured through \(k=12\); and
3. the complete A/B/C prompt-arm set is rerun on all three models.

Because earlier v8 results were inspected before this amendment, the replacement
must be described as an amended full rerun rather than an untouched prospective
preregistration.

## Claim and comparison

The experiment retains the two components of the v8 C2 criterion:

1. **C2a, latent-structure recovery:** infer two recurring response patterns
   from Cities A and B, then combine that structure with sparse City C evidence.
2. **C2b, contextual mechanism induction:** respond selectively to a
   mechanism-relevant target description and override a misleading description
   as City C evidence accumulates.

The main curve compares a City-C-only response regression with an A/B/C
regression-mixture. Since reporting begins after one completed City C case, no
\(k=0\) fallback is plotted or collected in this replacement.

## Frozen world and sample

- 120 episodes, seeds 80000--80119.
- Every row is an independent polling case beginning at 50.0 with net news
  \(+8\).
- The two hidden response patterns are 1.00 and 0.25 poll points per news point.
- Case noise has evaluator-only standard deviation 3.25.
- Each reference city shows six representative cases.
- City C contains twelve cases from the same seeded stream. For each seed,
  cases 1--8 are unchanged from the earlier v8 generator and cases 9--12 extend
  that stream.
- Prefixes are every integer \(k\in\{1,2,\ldots,12\}\).
- Target type, which A/B label is the high-response city, and three context
  paraphrase families are exactly balanced.

The model is never shown the equation, response values, noise scale, prior
probability, hidden type, evaluator match label, or context reliability.

## Prompt arms A, B, and C

All numerical content, city descriptions, sampling language, forecast question,
reasoning instruction, response schema, temperature, and token budget are
identical across arms.

- **A -- structure-blind:** no added structural instruction.
- **B -- relevance hint:** adds exactly:

  > When forecasting City C, consider whether the earlier cities' response
  > patterns and background descriptions are informative.

- **C -- strong relevance hint:** adds exactly:

  > Cities in this task follow two recurring response patterns, demonstrated by
  > Cities A and B. City C follows one of those patterns. Use City C's completed
  > cases and any relevant background information to decide how much weight to
  > place on each pattern when forecasting.

## Main C2a structure curve

The main curve uses the no-target-background condition. The two prompt-visible
comparisons remain:

- **Use City C only:** through-origin response regression on completed City C
  cases.
- **Combine A/B/C:** fit one response regression to A and one to B, estimate a
  residual scale from those displayed records, begin City C at equal weight over
  the two fitted patterns, and update the weights using City C's displayed cases.

The sparse anchor remains \(k=2\). The endpoint is now \(k=12\), and all twelve
integer steps are reported.

## Separate C2b context panel

Each numeric episode, arm, and prefix is rendered with no target background,
length-matched orthogonal prose, a high-transmission description, and a
buffered/low-transmission description. High and low descriptions remain crossed
with truth.

The context panel reports relevant movement, orthogonal movement, cue direction,
misleading-cue displacement, and the paired reduction in misleading-cue
displacement from \(k=1\) to \(k=12\).

## Models, inference, and collection size

The named models are Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4. They are
separate replications rather than a random model sample.

- Uncertainty intervals use deterministic 10,000-draw percentile bootstraps
  resampling episodes as clusters.
- Contrasts use only episodes with parsed responses in every required cell.
- One received completion is retained. An unparseable received completion is a
  terminal parsing failure; a transport failure before receipt may be retried.
- Each model-arm has \(120\times4\times12=5{,}760\) calls.
- Each model has 17,280 calls across A/B/C.
- The complete replacement has 51,840 calls across all three models.
- Figures show all three models on the same prompt-arm facets.
