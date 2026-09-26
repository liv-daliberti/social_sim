# Coin City structural and domain transfer

This experiment asks whether reward on Coin City's direct response transfers to
a new surface domain (Coin Harbor), an unseen mediated/persistent structure, or
both. Training prompts are byte-identical between the episode-matched `causal`
arm and the `population_prior` control; only the prompt-to-reward association is
broken in the control while the target marginal is preserved.

## Registered design and completed roster

- Models: Qwen3-4B-Instruct-2507, Qwen3-8B, and Llama-3.1-8B-Instruct.
- Confirmatory training: two arms x three models x seeds 42/43/44 (18 jobs).
- Diagnostic training: three Qwen3-4B structureless seeds.
- Untrained reference: one endpoint per model.
- Endpoint grid: 2 domains x 2 structures x 3 cues x 4 evidence depths x
  30 episodes = 1,440 tasks per checkpoint.
- Reported decoding: five draws at temperature 0.7.
- Monitoring provenance: one temperature-zero output per endpoint, excluded
  from manuscript estimates.

All six canaries, the fail-closed gate, and all 24 scientific endpoints are
complete. `reports/registered_results.json` is the canonical exact-roster
analysis: it validates 48 endpoint files and 207,360 scored draws before writing
any result. Its score-file, ledger, and analysis-code checksums provide the
release provenance.

The primary joint Coin Harbor/unseen-structure contrast is population-prior
minus episode-matched response MAE, positive in favor of episode-matched
training. Under five-draw decoding it is 0.319 [0.134, 0.489] for Qwen3-4B,
0.251 [0.058, 0.470] for Qwen3-8B, and 0.420 [-0.115, 0.958] for Llama-8B. The
paper therefore treats the two Qwen results as a scale replication and the
Llama training-effect estimate as inconclusive. Llama's stochastic cue-use and
evidence-override intervals exclude zero, providing conclusive evidence that it
uses the supplied mechanism cue and revises away from a misleading cue as target
observations accumulate. Temperature-zero endpoints remain audit artifacts, not
manuscript results.

## Registered Qwen3-14B scale extension

`QWEN3_14B_SCALE_PROTOCOL.md` freezes a post-original, prospectively reported
same-release 8B-to-14B scale test. The extension adds only Qwen3-14B: two reward
arms by seeds 42/43/44 plus one untrained base endpoint. It reuses the exact
1,440-task universe and omits the already-resolved structureless diagnostic.
One excluded ten-round canary gates every scientific submission.

The only submission interface is:

```bash
python exp3_training_transfer/coin_city_structural/launch_qwen3_14b_extension.py --stage canary
```

The launcher registers an `afterok` advance automatically. The advance audits
the canary, submits the exact six-training/one-base roster, and dependency-locks
the hash-pinned renderer and paper build behind all seven scientific jobs.
Confirmatory 14B training uses `account=allcs`, `partition=cs`, and two
A6000s; this pre-training allocation amendment changes no scientific setting.

## Maintained interfaces

From the repository root, run lightweight protocol and analysis regressions:

```bash
python -m pytest -q exp3_training_transfer/coin_city_structural/tests
```

Inspect registered scheduler state without modifying it:

```bash
python exp3_training_transfer/coin_city_structural/monitor.py \
  --ledger exp3_training_transfer/coin_city_structural/runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json
```

Rebuild the complete result and the four canonical paper tables:

```bash
python exp3_training_transfer/coin_city_structural/make_paper_outputs.py \
  --ledger exp3_training_transfer/coin_city_structural/runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json
```

The renderer fails on an incomplete/duplicate roster, a missing or ambiguous
temperature-zero or stochastic endpoint, identity drift, row-count drift, or a
different task universe. Training jobs store round-300 monitoring scores under
`debug_*/eval_results/301.scores.jsonl`; base jobs store `greedy.scores.jsonl` at
their report root. That legacy filename is retained for provenance. Both layouts
are resolved explicitly and tested.

The immutable launch and repair ledgers remain in `runs/`. Detailed pre-result
scheduler notes were moved to `_archive/status_snapshots/`; they are provenance,
not current operating instructions. The only maintained manuscript root is
`paper/`.
