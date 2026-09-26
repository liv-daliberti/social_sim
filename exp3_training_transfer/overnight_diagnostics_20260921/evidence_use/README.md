# Mechanism-family evidence-use follow-up

The registered design is in `PROTOCOL.md`. All frozen data, source and checkpoint
hashes are in `data/frozen_manifest.json`; eight real adapter files (seeds45/46,
matched/prior, both disclosures) and twelve missing metadata-only checkpoints are
listed in `data/checkpoint_inventory.json`. Original experiments are untouched.

Submitted GPU inference jobs are31444965 (disclosed) and31444966 (undisclosed).
`submission.json` preserves initial submission hashes and the resource correction.
Both jobs evaluate base and four available adapters with one base-model load:
384 paired-gain prompts and720 fresh ordinary prompts per checkpoint/disclosure.

Run from the repository root:

```bash
source .runtime/oat_env.sh
.runtime/oat_conda/bin/python exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/test_evidence.py
.runtime/oat_conda/bin/python exp3_training_transfer/overnight_diagnostics_20260921/evidence_use/analyze.py
```

Inference can resume complete checkpoint/suite outputs after verifying frozen
hashes. Submit `run_inference.sbatch` with `--export=ALL,DISCLOSURE=disclosed` or
`undisclosed`. Do not rebuild frozen datasets in place. The runner aborts if
prompt truncation, a changed adapter, or changed dataset is detected.

`analysis/RESULTS.md` and `analysis/summary.json` are generated as jobs finish.
The primary metric compares high-minus-low changes in shock-versus-zero-shock
responses with the simulator. Lower change MAE is better; tracking slope1 is the
simulator and0 is no reaction. Each side failing strict parsing makes its whole
pair a retained zero-change prediction. Base/matched/prior are displayed
separately; seed45/46 contrasts are exploratory, not five-seed evidence.

`known_mechanism_baseline_freeze.json` separately registers an auxiliary numerical
posterior reference before model predictions were inspected. It gets the true
mechanism even for undisclosed tasks, so it assesses observational adequacy rather
than forming a fair model comparator. Its results are in
`analysis/known_mechanism_baseline.json`. The extrapolation world's narrow gain
support and high observation noise limit its recoverable gain-change slope to
about0.13 in these pairs, while the other heldout worlds reach0.81–0.92.

The720-row ordinary heldout HF datasets `data/heldout_disclosed` and
`data/heldout_undisclosed` are also used by `../shuffled_control`; that experiment
owns its stochastic primary endpoint evaluations. These inference jobs use greedy
decoding under the fixed evidence-use protocol.
