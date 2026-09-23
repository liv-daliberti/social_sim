# Experiment 3/4 status and review handoff

Last scientifically audited: **2026-09-23 16:30 ET**. The prior audit was
2026-08-24; a month of drift between those two dates is how the Qwen3-14B
rendering below went unnoticed, so re-audit this file whenever a campaign
changes state. Scheduler state changes
more quickly than this document; use the read-only monitors below for job-level
status. The superseded August 20 snapshot is preserved under
`_archive/status_snapshots/`.

## Durable status

| Campaign | Scientific status | Paper status |
|---|---|---|
| Structural-OOD mechanism grid | Complete: exact 42-training/6-base roster | Full aggregate and four tables generated |
| Coin City structural/domain transfer | Complete: 24 exact-roster endpoints after six canaries and a gate | Stochastic Qwen3-8B main result; stochastic Qwen3-4B/Llama-8B appendix comparators |
| Coin City Qwen3-14B scale extension | **Blocked, not complete.** Five of six trainings completed 2026-08-26. Causal seed 43 (`30880156`) hit its 30h walltime; its rerun `30920615` produced no scores; a third run `30948628` produced scores but under a modified `train.sh`. Retrain `31481143` submitted 2026-09-23 under the registered script with a 2-day allocation. | **Nothing rendered.** Do not render until the retrain lands and a ledger naming the real jobs is committed. |
| Historical Polymarket Qwen3-4B | Complete: three training seeds plus sealed test | Appendix scale comparator and secondary update analysis |
| Polymarket Qwen3-8B/Llama-8B extension | Complete: six replacement trainings and two locked tests | Qwen3-8B main result; Llama-8B appendix comparator |

The structural aggregate
`mechanism_family/reports/c3_mechanism_full_aggregate.json` records 207,360
endpoint rows (414,720 analysis rows including overall duplicates), the exact
ledger, all 96 score-file hashes, and all three analysis-code hashes. The August
23 review audit verified all 100 recorded hashes.

The Coin City renderer validates 18 confirmatory, three diagnostic, and three
base endpoints; temperature-zero monitoring and five-draw files must contain
the identical 1,440-task universe. The manuscript reports only the five-draw
stochastic endpoints. Its canonical result is
`coin_city_structural/reports/registered_results.json` (207,360 scored draws).
The full renderer now resolves training jobs' registered round-300 greedy files
from their runtime directories and rejects missing or ambiguous endpoints.

The post-original Qwen3-14B Coin City extension is frozen under protocol
`coin_city_qwen3_14b_scale_v1`. Canary `30872875` and advance `30872876`
completed successfully. Canary greedy and stochastic parse rates were both
100%; base endpoint `30879415` also completed with full parse coverage. Six
never-started A100 submissions (`30879409`--`30879414`) were superseded before
training. Their replacements (`30880155`--`30880160`) request two A6000s each
under `allcs`/`cs` and are eligible, waiting on scheduler priority. Finalizer
`30880231` remains gated on all six trainings.

## Open items as of 2026-09-23

**Qwen3-14B Coin City extension: blocked, and the registration record is not
trustworthy as it stands.** Three separate problems, all of which must be fixed
before anything is rendered:

1. The registered `train_script_sha256` no longer matches `train.sh`. The
   working-tree script gained periodic checkpointing on 2026-08-28 so a
   timed-out training could be recovered. The change is benign -- defaults
   reproduce the registered invocation, and the recovered run trained from
   scratch (`resume_dir` empty) to the registered `step 301` -- but it is
   uncommitted, and only causal seed 43 ran under it.
2. The ledger names `30920615` for causal seed 43. That job produced no scores.
   The scores live under `30948628`, which the ledger does not name.
3. The ledger `runs/coin_city_qwen3_14b_scale_full_20260825T134340Z.json` is
   untracked and its mtime is 2026-08-27, two days after the timestamp in its
   own filename. It was edited after registration with no version history.

Rendering through these would mean accepting a drifted script, substituting an
unregistered job, and trusting a hand-edited record -- not a checksum waiver.
The fix in progress instead: job `31481143`, submitted 2026-09-23, retrains
causal seed 43 from `train_registered_s43.sh`, which is byte-identical to the
committed registered script, with a 2-day allocation because the failure was
walltime and walltime is an `sbatch` parameter rather than part of the script.
When it lands, write a ledger naming the real jobs and **commit it** before
rendering.

**The `sel_` runs are the structure-selection experiment, and they are already
in the paper.** Corrects an earlier reading in this file that treated them as
unregistered. They belong to `coin_city_structure_selection/`, protocol
`coin_city_structure_selection_v1`, frozen 2026-09-16 before any training job
existed, and they are reported in App. `app:exp3-structure-selection`. They sit
outside `reports/registered_results.json` because that ledger belongs to the
parent `coin_city_structural_transfer_v1`, which this protocol explicitly does
not modify or supersede.

That sibling exists because the parent cannot answer the structure question. The
parent's training set varies gain, not structure, so "emit zero at horizon 3" is
optimal under its training distribution: measured horizon-3 output is exactly
0.000 in every evaluation cell on every seed in both arms, and the cue moves
structure choice by -0.003 to +0.001 against Experiment 2's .51-.86. The paper
states this.

Compute is complete: `sel_causal_qwen3_8b` s42-44, `sel_population_prior_qwen3_8b`
s42-44 and `selbase_qwen3_8b` all finished 2026-09-20/21. Every Exp 3 macro in
`paper/tables/exp3_rebuilt_data.tex` regenerates byte-identically from
`render_exp3_paper_macros.py`, so the manuscript is current against everything on
disk.

Two genuine decisions remain, both narrower than "does this belong":

1. The appendix reports a three-seed pilot of the episode-matched arm and
   declines the episode-matched minus population-prior contrast, saying it needs
   the eight-seed treatment. The population-prior arm now exists at three seeds,
   so the contrast is computable but still below the bar the protocol set.
   Extend to eight seeds, or leave the pilot as written.
2. `causal`/`population_prior` `llama3_1_8b` s45-46 completed 2026-09-21. The
   appendix says the Llama-8B comparison "remains a three-seed analysis"; these
   would make it five.

**Seven-seed contrasts are computing, deliberately unpublished.** The seed 47/48
trainings are still running with a finalizer queued behind them.
`freeze_extension.json` states that seven-seed estimates are reported alongside,
never in place of, the frozen five-seed roster, so whether the appendix gains a
sentence remains an author decision.

**Paper wiring, resolved 2026-09-23.** Four three-seed outputs had been generated
but never included: `exp3_coin_structural_summary`, `exp3_coin_structural_cues`,
`exp3_coin_structural_diagnostics`, and `exp3_coin_qwen3_4b_stochastic`. All four
now appear in the three-seed subsection of the appendix. Four others are
superseded and should stay unwired: `exp3_coin_qwen3_4b_cue_by_k` and
`exp3_coin_qwen3_8b_cue_by_k` duplicate `exp3_coin_structural_cues` per model;
`exp3_coin_qwen3_8b_stochastic` is superseded by the eight-seed data file used in
the main text; and `exp3_coin_qwen3_4b_stochastic_data` only defines `\cciPeak`
for a Qwen3-4B figure that does not exist, and would collide with the Qwen3-8B
definition if included.

## Scheduler lineage and paper handoff

The original Polymarket 8B jobs failed before scientific training because actor
workers lacked the shared mechanism-family directory on `PYTHONPATH`. Their
scientifically identical replacement roster and failure marker are recorded in
`polymarket/runs/exp3b_model_extension_20260824T012515Z.json`.

All replacement jobs completed successfully. Qwen3-8B improves from .12867 to
.11642 Brier (trained-minus-base $-.01225$ $[-.01948,-.00602]$); Llama-8B
improves from .15203 to .11502 ($-.03701$ $[-.05376,-.02258]$). Both
trained-minus-market intervals span zero. Both trained means have higher error
than the train-only Platt market baseline.

The old Coin City paper job `30788375` failed before rendering because its
submission-time hash pin predated the single-manuscript cleanup; terminal job
`30788376` therefore cannot run. This was a handoff failure, not an endpoint or
analysis failure. The repaired renderer has since validated the complete roster
and generated the canonical result and tables directly. Combined finalizer
`30856298` completed mechanism aggregation but failed in the paper renderer
because its raw-row validator rejected the 12 registered, penalized parse
failures in the untrained Llama endpoint. The locked summaries and scientific
jobs were unaffected. The validator now accepts `null` only under the registered
loss-one policy, recomputes coverage and Brier from raw rows, and writes both
tables into the sole physical manuscript at `paper/`.

## Read-only checks

```bash
python exp3_training_transfer/monitor_campaign.py \
  --mechanism-ledger exp3_training_transfer/mechanism_family/runs/c3_mechanism_20260817T215355Z_walltime30h_20260818T211828Z_runtimefix_20260819T172806Z.json \
  --polymarket-extension-ledger exp3_training_transfer/polymarket/runs/exp3b_model_extension_20260824T012515Z.json

python exp3_training_transfer/coin_city_structural/monitor.py \
  --ledger exp3_training_transfer/coin_city_structural/runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json
```

Repository review and manuscript reconstruction remain:

```bash
make review
make paper
```
