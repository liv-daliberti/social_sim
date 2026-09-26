# Experiment 3A: registered run status

_Last updated 2026-08-12._

## Completed

- Family and structureless-control training finished at step 300 for seeds 42, 43,
  and 44.
- The registered preflight passed for all three arms: 4,800 paired training rows
  per arm, 1,440 held-out rows, disjoint train/evaluation seeds, identical
  family/prior-mean prompts, and identical scored held-out content after removing
  the prior-mean provenance field.
- On training rows, `corr(g, slope_target)` is 0.8091 for family and -0.0039 for
  prior mean. The stronger control therefore removes per-city gain information
  while retaining the coherent input distribution.

Authoritative audit: `protocol/exp3a_manifest.json`.

## Running

The first prior-mean submission failed before evaluation because Launchpad workers
could not import `forecast_scoring`; the parent processes then waited until their
wall-time limits. The worker import path is repaired and validated from OAT's
runtime directory. Replacement jobs are submitted for the registered seed triplet:

| Seed | Slurm job | Current state |
|---:|---:|---|
| 42 | 30497077 | pending (priority) |
| 43 | 30497078 | pending (priority) |
| 44 | 30497079 | pending (priority) |

Submission ledger: `runs/exp3a_prior_mean_20260812T134459Z.json`.

## After the jobs finish

- Run the fail-closed endpoint report after 300 updates for all three paired arms.
- Regenerate `exp3_transfer.tex` and the transfer figures from saved dumps.
- Treat family minus prior mean as the per-city-inference contrast. Family minus
  structureless only tests whether training used forecast structure at all.
- Report each held-out world separately. Do not pool raw response errors whose
  scales differ, and do not interpret the structureless `chain4` error contrast;
  its zero-slope target is accidentally close to that world's small mean response.

## Remaining limitations

- `chain4` is a low-power recovery probe: its first-response signal is small
  relative to observation noise.
- The current headline covers Qwen3-4B, 300 updates, and three seeds. The legacy
  scale-ladder outputs use an obsolete probe and remain excluded.
- A wrong-structure control would test a distinct question and is not part of the
  registered three-arm comparison.
