# Experiment 4: historical Polymarket transfer

This directory contains the registered historical-market training experiment.
`PROTOCOL.md` specifies the original experiment; `MANIFEST.md` and
`docs/DATASET_CARD.md` document the upstream archive. `SCALE_PROTOCOL.md`
specifies the family-disjoint scale holdout, and
`ARCHITECTURE_PROVIDER_SWEEP_PROTOCOL.md` specifies the post-hoc local-model
and hosted-provider extension.

## Review-facing result hierarchy

The original leakage-controlled split contains 1,736 train, 512 development,
and 1,024 sealed-test markets. Event IDs, available series IDs, and normalized
question templates are disjoint across splits. Those endpoint and architecture
results remain in the appendix.

The main-body scaling figure uses the later 318-market family-disjoint holdout.
It connects only the Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-14B three-seed
results. Llama-3.2-3B and Llama-3.1-8B are separately registered, unconnected
architecture comparators. Every local endpoint uses five-draw non-thinking
stochastic decoding. The main
comparison is held-out Brier score against the contemporaneous Polymarket crowd;
the train-only Platt baseline and the old controlled 14B-versus-8B framing are
not part of the main-body presentation.

## Maintained interfaces

Key entrypoints are:

- `scripts/build_exp3b_dataset.py`: construct the original registered split;
- `scripts/preflight_exp3b.py`: fail-closed split, prompt, hash, and token audit;
- `scripts/launch_registered.py`: original registered training launcher;
- `scripts/evaluate_locked_test.py`: original sealed-test evaluator;
- `scripts/evaluate_exp1_reapplication.py` and
  `scripts/analyze_exp1_reapplication.py`: fixed-initial-forecast update study;
- `scripts/launch_model_extension.py` and
  `scripts/evaluate_locked_test_model_extension.py`: original
  Qwen3-8B/Llama-3.1-8B appendix extension;
- `scripts/launch_scale_extension.py`: registered Qwen scale launcher;
- `scripts/render_scale_extension.py`: validated scale summary renderer;
- `scripts/polymarket_rl_architecture_extension.sh`: registered
  Qwen3-1.7B/Llama-3.2-3B training wrapper;
- `scripts/locked_test_architecture_extension.sbatch`: dependency-locked
  318-market evaluator for the new local checkpoints;
- `scripts/finalize_architecture_extension.py`: fail-closed artifact, task,
  adapter, decoder, and job-lineage gate before paper ingestion;
- `scripts/evaluate_hosted_scale_holdout.py`: append-only hosted-provider shard
  runner and finalizer;
- `scripts/evaluate_hosted_scale_holdout_opus5.py`: deployment-specific Opus 5
  adapter over the same hosted runner; and
- `../../paper/generate_exp4_scale_figure.py`: fail-closed paper figure/table
  generator for completed local results.

Run the lightweight review gate from the repository root:

```bash
python -m pytest -q exp3_training_transfer/polymarket/tests
```

The full preflight loads the frozen local tokenizer and dataset cache, so it is
intentionally separate from the lightweight gate.

## Completed result lineages

The original three-model roster is recorded in
`runs/exp3b_model_extension_20260824T012515Z.json`. All replacement jobs reached
300 rounds, both dependency-locked tests completed, and the renderer validates
lineage, task identity, parse coverage, failure penalties, and frozen summaries.

The 318-market scale replication is frozen by `SCALE_PROTOCOL.md`. Its holdout
excludes every event, series, normalized template, and task used by the original
train, development, or test splits. Qwen3-4B, Qwen3-8B, and Qwen3-14B are shown
as one solid same-family line. The separately registered Llama-3.1-8B result is
shown with square markers and is never connected to that line.

## Active architecture and provider sweep

The post-hoc sweep was registered before any new training or hosted holdout call
in
`runs/exp4_architecture_provider_sweep_registration_20260825T203304Z.json`.
It adds:

- `Qwen/Qwen3-1.7B` to the solid Qwen family line;
- the pinned public BF16 `unsloth/Llama-3.2-3B-Instruct` mirror as a second
  unconnected Llama point; and
- `claude-opus-4-8`, `gpt-5.6-sol`, `FW-Kimi-K3`, and
  `DeepSeek-V4-Pro` as a separate off-the-shelf API reference panel.

The later Opus 5 addition is separately registered, after the preceding results
were observed, in
`runs/exp4_architecture_provider_sweep_opus5_registration_20260826T143622Z.json`.
Its result is retained in the appendix but excluded from the main categorical
panel. Kimi's user-authorized holdout-informed repair is marked with an asterisk;
the original and intermediate fail-closed artifacts remain archived.

The local checkpoints use the same 1,736/512 train/development data, three seeds,
300 updates, rank-32 LoRA, learning rate, stochastic rollout recipe, and sealed
318-market evaluator as registered. Scheduler changes are resource-only and
recorded under `runs/`. A development-only amendment permits at least 99%
parseable draws while requiring every task to have a parseable draw; final
holdout missingness remains fail-closed with task Brier loss one.

After both dependency-locked evaluations complete, the review handoff is:

```bash
python exp3_training_transfer/polymarket/scripts/finalize_architecture_extension.py
python paper/generate_exp4_scale_figure.py \
  --architecture-extension exp3_training_transfer/polymarket/reports/exp4_architecture_extension_latest.summary.json
```

The second command cannot ingest results that fail the first command's checks.

The hosted panel receives the exact local user-message strings, no tools or
search, and five independent calls per market. It shares the parser, aggregation,
failure penalty, market comparator, and bootstrap. Provider APIs do not expose
identical decoder controls, so hosted results must remain in a separate panel
and must not be described as decoder-identical or trained-model results.
Development transport canaries use two development prompts per provider.
Sending any prompt to an external provider requires explicit authorization;
dry runs never make an API call.

Example no-network validation:

```bash
python exp3_training_transfer/polymarket/scripts/evaluate_hosted_scale_holdout.py \
  --model claude-opus-4-8 --split dev --limit 2 --draw-index 0 --dry-run
```

## Non-negotiable reporting invariants

- Use stochastic decoding everywhere; do not add a deterministic/greedy endpoint.
- Connect only checkpoints from the same Qwen family.
- Keep every Llama and hosted-system point visually unconnected from Qwen.
- Use the contemporaneous Polymarket crowd in the main figure; keep Platt in the
  appendix only.
- Report all registered results regardless of direction.
- Verify report, raw artifact, adapter, prompt, code, and registration hashes
  before rendering paper assets.

Raw multi-gigabyte snapshots and training reports are workspace artifacts, not
Git payloads. The anonymous release builder under `hf_release/` is the public
distribution interface.
