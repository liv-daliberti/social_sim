# Three-city C2 v8 local full k=1...15 matched-estimator specification

This amendment was frozen on 2026-08-04 before the matched local collection was
launched. It supersedes the queued k=1...12 replacement, whose Slurm jobs were
cancelled while still pending and therefore made no model calls. A first local
k=1...15 launch was stopped on 2026-08-04 after its asymmetric estimator
comparison was rejected. Its partial responses are archived under
`full_k1_15_asymmetric_aborted_20260804` and are not pooled with this run. The
earlier stopped n=50 collection and k=1...12 preview also remain separate.

Because earlier v8 results and the first local partial collection were inspected
before this amendment, this is an amended full rerun rather than an untouched
prospective preregistration.

## Frozen changes

1. Report every integer prefix from \(k=1\) through \(k=15\).
2. Rerun the complete A/B/C prompt-arm set on all three named models.
3. Execute all nine shards directly on the login node rather than through Slurm.
4. Compare matched continuous estimators, so A/B changes finite-sample
   regularization but not the asymptotic target of the City C estimate.

## Frozen world and sample

- 120 episodes, seeds 80000--80119.
- Every row is an independent polling case beginning at 50.0 with net news
  \(+8\).
- Hidden response patterns are 1.00 and 0.25 poll points per news point.
- Evaluator-only case noise has standard deviation 3.25.
- Each reference city shows six representative cases.
- City C contains fifteen cases from the same seeded stream.
- Prefixes are every integer \(k\in\{1,2,\ldots,15\}\).
- Target type, reference label, and three context families are balanced.

## Prompt arms A, B, and C

All task data and common instructions are identical across arms.

- **A -- structure-blind:** no structural instruction.
- **B -- relevance hint:** asks the model to consider whether earlier-city
  response patterns and descriptions are informative.
- **C -- continuous-pooling hint:** says A/B may be related but need not be
  identical to C, and directs the forecaster to give C increasing weight.

The C text replaces the earlier exact-pattern hint. The A and B arms retain
their prior wording.

## Matched structure outcomes

The no-target-background structure curve compares:

- **Use City C only:** a through-origin response regression on completed C cases.
- **Combine A/B/C (matched):** an empirical-Bayes posterior mean for the same
  continuous City C response coefficient. The fitted A/B slopes provide a
  Gaussian prior mean and unbiased sample variance. Displayed A/B residuals
  provide the likelihood variance. Completed C cases provide the same
  through-origin regression used by the C-only estimator.

The A/B prior weight decreases mechanically with every added C case and tends
to zero. Thus A/B/C can reduce early sampling error, but it cannot collapse onto
an exact hidden class while C-only continues to estimate a noisy mean. On the
frozen seeds, the baseline MAEs are 2.788 (C only) versus 2.047 (A/B/C) at
\(k=1\), and 0.594 versus 0.588 at \(k=15\). The nonzero common tail is ordinary
finite-sample noise, not a difference in estimator class.

The primary sparse anchor remains \(k=2\), and every step is plotted through
\(k=15\). Exact-type and context-aware evaluator oracles remain diagnostic
ceilings only. Neither is labeled or plotted as the matched A/B/C baseline.

The context panel continues to report relevant movement, orthogonal movement,
cue direction, misleading-cue displacement, and paired override from \(k=1\)
to \(k=15\).

## Models, execution, and size

Models are Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4. All nine model-by-arm
shards run concurrently as background processes on the login node. No Slurm job
is used. Responses, logs, process records, caches, and live figures are stored
under `data/three_city_c2_v8/full_k1_15`.

- Each model-arm has \(120\times4\times15=7{,}200\) calls.
- Each model has 21,600 calls across A/B/C.
- The complete run has 64,800 calls.
- Uncertainty intervals use deterministic 10,000-draw episode bootstraps.
- A received but unparseable completion is terminal. A pre-completion transport
  failure remains retryable on a later invocation.
