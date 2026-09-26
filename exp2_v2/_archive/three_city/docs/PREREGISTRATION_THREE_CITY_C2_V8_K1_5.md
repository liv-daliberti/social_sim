# Three-city C2 v8 local full k=1...5 matched-estimator specification

This amendment was frozen on 2026-08-04 before the k=1...5 replacement
collection was launched. The prior matched k=1...15 local collection was stopped
at user request and archived under
`full_k1_15_matched_aborted_20260804`. Its partial responses are not pooled with
this run. The earlier asymmetric collection remains separately archived under
`full_k1_15_asymmetric_aborted_20260804`.

Because earlier designs and partial responses were inspected before this
amendment, this is an amended full rerun rather than an untouched prospective
preregistration.

## Frozen changes

1. Report every integer prefix \(k\in\{1,2,3,4,5\}\).
2. Make no calls for prefixes above \(k=5\).
3. Rerun all three prompt arms on Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4.
4. Execute all nine shards directly on the login node.
5. Retain the matched continuous estimator comparison.

## Frozen world and sample

- 120 episodes, seeds 80000--80119.
- Every row is an independent polling case beginning at 50.0 with net news
  \(+8\).
- Hidden response patterns are 1.00 and 0.25 poll points per news point.
- Evaluator-only case noise has standard deviation 3.25.
- Each reference city shows six representative cases.
- City C contains five cases from the original seeded stream.
- Target type, reference label, and three context families are balanced.

## Prompt arms A, B, and C

All task data and common instructions are identical across arms.

- **A -- structure-blind:** no structural instruction.
- **B -- relevance hint:** asks the model to consider whether earlier-city
  response patterns and descriptions are informative.
- **C -- continuous-pooling hint:** says A/B may be related but need not be
  identical to C, and directs the forecaster to give C increasing weight.

## Matched structure outcomes

The no-target-background structure curve compares:

- **Use City C only:** a through-origin response regression on completed C cases.
- **Combine A/B/C (matched):** an empirical-Bayes posterior mean for the same
  continuous City C response coefficient. Fitted A/B slopes provide a Gaussian
  prior mean and unbiased sample variance. Displayed A/B residuals provide the
  likelihood variance. Completed C cases provide the same through-origin
  regression used by the C-only estimator.

The A/B prior weight decreases at every step. On the frozen seeds, baseline MAE
is 2.788 (C only) versus 2.047 (A/B/C) at \(k=1\), and 1.010 versus 0.988 at
\(k=5\). Mean A/B weight falls from 0.355 to 0.103 across those steps. Thus the
matched curves are already close at the requested endpoint without granting the
A/B/C estimator an exact-classification shortcut.

The primary sparse anchor remains \(k=2\). Exact-type and context-aware
evaluator oracles remain diagnostic ceilings only and are not labeled as the
matched A/B/C baseline.

The context panel reports relevant movement, orthogonal movement, cue direction,
misleading-cue displacement, and paired override from \(k=1\) to \(k=5\).

## Models, execution, and size

All nine model-by-arm shards run concurrently as background processes on the
login node. No Slurm job is used. Responses, logs, process records, caches, and
live figures are stored under `data/three_city_c2_v8/full_k1_5`.

- Each model-arm has \(120\times4\times5=2{,}400\) calls.
- Each model has 7,200 calls across A/B/C.
- The complete run has 21,600 calls.
- Uncertainty intervals use deterministic 10,000-draw episode bootstraps.
- A received but unparseable completion is terminal. A pre-completion transport
  failure remains retryable on a later invocation.
