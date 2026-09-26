# Experiment 3 latent-mechanism probe

This appendix-only diagnostic asks whether Experiment 3 training makes an
episode's latent response mechanism more linearly available at the end of the
prompt. It compares the frozen base model with the `causal_family` and
`population_prior` LoRA endpoints from the completed structural-OOD grid. A
linear probe is correlational; it does not establish that a decoded direction
causes the forecast.

## Frozen task and split

The source is the existing 720-row structural-OOD held-out set. Its four unseen
worlds each contain 60 numeric episodes rendered at `k in {3, 6, 9}`. Every
numeric task is paired across disclosed and undisclosed prompts. The probe task
builder assigns 45 episodes per world to development and 15 to sealed test,
using only the pre-existing episode order. All three evidence depths and both
disclosure arms for an episode remain in the same split. Five development folds
contain nine episodes per world each.

Every checkpoint is evaluated on both disclosure arms. This permits a matched
cross-disclosure test even when an adapter was trained under only one disclosure
condition. No Experiment 3 training row enters probe fitting.

## Targets and controls

The primary targets deliberately remove information available from surface
labels or the disclosed mechanism paragraph:

1. **Within-world gain**: episode-specific `g`, centered and scaled using only
   development episodes within each world.
2. **Within-world impulse response**: the eight nonzero-shock contrasts after
   subtracting the development mean separately by world and evidence depth.

World/block classification is not a primary target because each held-out world
has a unique domain name and description. Decoding it would mostly measure
surface recognition. The primary representation is the final input-token state
at every layer. Mean-pooled embedding and final-layer states are frozen pooling
controls.

For every disclosure/evidence-depth/target cell, five-fold episode-balanced
development CV selects ridge strength at each layer and then selects one layer.
The sealed test is read once. The same fitted probe is also evaluated on the
matched opposite-disclosure prompts. A within-world held-out permutation test
keeps the selected layer, ridge strength, and fitted weights fixed.

## Endpoint roster

The pilot is Qwen3-8B base plus seed-42 `causal_family` and
`population_prior` adapters trained under each disclosure (five extraction
jobs). The replication roster is Qwen3-4B-Instruct-2507 and
Llama-3.1-8B-Instruct over the same five endpoints. A training-effect claim
requires the later seed-43/44 replication; the seed-42 pilot alone is
diagnostic. Results stay out of the paper until a fail-closed aggregate requires
the registered endpoints and all extraction hashes.

## Commands

Build and audit the deterministic 1,440-prompt task file:

```bash
.runtime/oat_conda/bin/python \
  exp3_training_transfer/mechanism_family/mechanistic_probe/build_probe_tasks.py
```

Render the five Qwen3-8B pilot submissions without changing scheduler state:

```bash
python exp3_training_transfer/mechanism_family/mechanistic_probe/launch_probe.py
```

Submit the pilot only after reviewing that output:

```bash
python exp3_training_transfer/mechanism_family/mechanistic_probe/launch_probe.py --submit
```

The launcher resolves each registered adapter to exactly one final
`step_00301`; missing or ambiguous checkpoints fail closed. Use `--models
qwen3_4b llama3_1_8b` to render the replication roster.

After extraction, each job runs `analyze_probe.py` with 10,000 held-out
permutations and writes `probe_results.json`, `layerwise_results.csv`, and an
extraction manifest under `mechanistic_probe/runs/`.
