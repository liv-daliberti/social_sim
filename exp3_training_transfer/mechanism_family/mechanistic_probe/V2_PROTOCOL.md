# Experiment 3 latent-mechanism probe v2

Protocol version: `exp3_mechanism_probe_v2`

This version was defined after inspection of the seed-42 Qwen3-8B pilot and
before extraction of seeds 43/44 or either cross-model replication roster.  It
preserves the frozen 1,440 prompts, episode split, model checkpoints, and v1
artifacts.  V2 addresses analysis limitations revealed by the pilot and writes
new filenames rather than overwriting v1.

## Confirmatory scope

The generalization claim requires the complete roster:

- Qwen3-4B-Instruct-2507, Qwen3-8B, and Llama-3.1-8B-Instruct;
- untrained base plus causal-family and population-prior adapters;
- disclosed- and undisclosed-training adapters at seeds 42, 43, and 44; and
- evaluation of every endpoint on paired disclosed and undisclosed prompts.

This is 13 endpoints per architecture (39 total).  Five Qwen3-8B seed-42
extractions were completed for the pilot, leaving 34 extraction jobs at the
time v2 was frozen.  Training seed is the top-level replication unit for
causal-versus-prior and trained-versus-base inference.  Episode resampling is
nested within selected training seed.  Architecture is reported separately;
the three architectures are not treated as exchangeable random draws.

## Primary targets and leakage-safe centering

The primary targets remain episode gain and the truth-minus-frozen-prior
eight-dimensional impulse response.  For every development fold, target means
are estimated using only the fitting episodes within held-out world.  The fold
validation targets are residualized by those fitting means.  The sealed test
is residualized only by the full-development means.

The primary score is held-out R2 relative to the development-estimated world
mean.  The conventional R2 relative to the sealed test's oracle within-world
mean is retained as a secondary descriptive score.  Both are reported to make
finite-sample development/test mean drift explicit.

## Pooled evidence-depth probe

In addition to the original separate `k=3,6,9` cells, a pooled probe fits all
evidence depths with episode-grouped folds.  Both targets and hidden features
are residualized using fitting-only world-by-k means.  This treats world and
evidence depth as nuisance factors while increasing development rows from 180
to 540; all three presentations of an episode remain in the same fold.

## Numerical positive controls

A sidecar keyed by frozen sample ID supplies prompt-derived controls without
changing prompt hashes or hidden-state order:

- last displayed target observation;
- mean displayed target observation;
- displayed target time slope;
- contemporaneous displayed driver/outcome slope; and
- prompt-information posterior mean gain.

For disclosed prompts the posterior uses the displayed mechanism and gain
range.  For undisclosed prompts it marginalizes the frozen mechanism catalog
using the displayed calibration and target trajectories.  Failure on these
controls limits claims about the final-token anchor's numerical accessibility.

## Cross-disclosure transfer

V2 retains direct zero-shot cross-disclosure transfer and adds an unlabeled
mean-aligned test.  For each development world (world-by-k in the pooled
analysis), paired opposite-disclosure prompts estimate a feature translation.
That translation is applied to sealed opposite-disclosure prompts before the
unchanged source probe is evaluated.  A large improvement after translation
indicates disclosure-specific affine shift rather than absence of a shared
readout direction.

Cross-disclosure results are secondary.  They are not used to claim invariance
or non-invariance unless same-disclosure decoding first exceeds its
development-estimated baseline.

## Output-level positive control

Registered five-draw generated responses are averaged within task and evaluated
on the identical episode split.  Development rows alone define within-world
response baselines and the response-to-gain calibration.  Reported quantities
are within-world centered correlation, response R2 and MAE improvement relative
to the development baseline, and held-out gain-calibration R2.  Episode
bootstraps use 10,000 repetitions.  An all-failed stochastic task is assigned
the frozen population-prior response, which contributes zero episode-specific
information; partial failures average the parsed draws and coverage is reported.

## Saved predictions and inference

Every selected readout writes row-level sealed predictions for same-disclosure,
direct cross-disclosure, and mean-aligned cross-disclosure evaluation.  These
records include task, episode, world, k, raw target, development baseline,
residual target, residual prediction, and reconstructed raw prediction.

Training-effect summaries must pair endpoints on the shared sealed tasks,
resample training seeds at the top level, and resample episodes within seed.
Individual cell permutation p-values remain descriptive and are not counted as
independent replications.  A broad claim that training creates or fails to
create a code requires consistent direction across Qwen sizes and the Llama
architecture comparator.

## Versioned outputs

- `probe_results_v2.json`
- `layerwise_results_v2.csv`
- `sealed_predictions_v2.jsonl`
- `output_tracking_v2.json`
- `output_tracking_predictions_v2.jsonl`

The original `probe_results.json` and `layerwise_results.csv` remain unchanged.
