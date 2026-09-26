# C3 paired structural-OOD mechanism benchmark

This benchmark expands C3 without changing C2. It asks two separate questions over the same
numeric episodes and the same train/test mechanism catalog:

1. **Disclosed structural OOD:** can training transfer to unseen graph/function compositions and
   extrapolated parameters when the held-out mechanism is described?
2. **Undisclosed structural OOD:** can training transfer when the model must infer that same held-out
   mechanism from three calibration trajectories and a sparse target trajectory?

The disclosed and undisclosed prompts differ only in the mechanism-description block. They are not
separate data-generating processes.

## Worlds and held-out blocks

Eight balanced training worlds expose individual primitives: direct effects, persistence,
mediation, feedback, a second driver, saturation, and asymmetric thresholds. Four held-out worlds
are native domains rather than affine skins of a fixed graph:

| Block | Held-out composition |
|---|---|
| `topology_composition` | mediated response plus feedback |
| `nonlinear_composition` | saturating response plus mediation |
| `mixed_composition` | asymmetric threshold plus persistence |
| `parameter_extrapolation` | saturating feedback with gain and noise outside the train hull |

Each task asks for ten counterfactual forecasts: five signed shocks at horizons one and three. The
behavioral response vector consists of the eight nonzero-shock minus zero-shock contrasts, so the
score reads sign, delay, persistence, saturation, and asymmetry without using a self-reported
coefficient.

## Registered factorial

The matched model roster is Qwen3-4B-Instruct-2507, Qwen3-8B, and
Llama-3.1-8B-Instruct. Every confirmatory cell uses LoRA rank 32, alpha 64, 4,800 training rows,
300 rollout-and-update rounds (2,400 AdamW steps), rollout count eight, and seeds 42/43/44.

| Component | Count |
|---|---:|
| 2 disclosures x 3 models x 2 confirmatory arms x 3 seeds | 36 training jobs |
| 2 disclosures x Qwen3-4B x structureless x 3 seeds | 6 diagnostic jobs |
| 2 disclosures x 3 untrained models | 6 base-evaluation jobs |
| Total synthetic grid | 48 jobs |

The confirmatory arms are `causal_family` and `population_prior`. Their prompts are byte-identical;
only reward targets differ. The prior arm receives the best frozen catalog-wide impulse-response
policy and cannot use the episode's mechanism or gain. `structureless` independently resamples all
displayed drivers and has a flat response target; it is diagnostic rather than the primary control.

## Evaluation and variance

Every checkpoint keeps the registered greedy evaluation and adds five endpoint draws at temperature
0.7. Primary outcomes are response MAE, response error normalized by the frozen prior floor,
response informativeness, and behavioral mechanism-signature accuracy, all reported separately by
held-out block and evidence depth. `report.py` pairs causal and prior episodes, averages repeated
decodes within episode, resamples episodes within training seed, and resamples the three training
seeds at the top level. Decode draws are never treated as independent training replications.

The primary paired contrast is population-prior minus causal-family response MAE, pooled equally
over all 720 held-out episodes. The same contrast within each of the four held-out blocks is a
prespecified stratified analysis. Secondary pre-result contrasts use the same hierarchy:
untrained base minus causal-family error,
structureless minus causal-family error, the difference in causal-vs-prior benefit for
Qwen3-8B versus Qwen3-4B and Qwen3-8B versus Llama-3.1-8B, and the disclosed-minus-undisclosed
difference in causal-vs-prior benefit. Positive values favor causal training or the first-named
condition.

All synthetic training, online evaluation, endpoint evaluation, and base evaluation use the same
syntax-only XGrammar GBNF constraint: exactly one `forecasts` array with ten JSON numbers. The
grammar constrains neither forecast values nor their relationship to the targets; the unchanged
strict parser and reward score the resulting numbers. There is no parser repair or model-specific
exception. This v4 decoding correction replaces the failed v3 canary and all six canary cells are
rerun from scratch before the full grid can unlock.

Synthetic training jobs request two A6000s, eight CPU cores, 100 GB host memory, and the registered
training wall time. Untrained-base jobs request one A6000, eight CPU cores, 24 GB host memory, and
a 30-minute wall-time limit.
## Reproduction

```bash
# Already generated for the registered run:
.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/make_dataset.py \
  --all --n-train-per 600 --n-eval-per 60

.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/preflight.py \
  --oracle-per-world 3 \
  --write-manifest exp3_training_transfer/mechanism_family/protocol/c3_mechanism_manifest.json

# Render the six canaries without submitting:
.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/launch_factorial.py \
  --canary --dry-run --allow-missing-model

# Submit only after all three model snapshots are present and canaries pass:
.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/launch_factorial.py --canary
.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/launch_factorial.py --full
```

The launcher refuses a real full submission unless the data manifest records exactly 600 training
episodes per training world and 60 evaluation episodes per held-out world. It reruns the fail-closed
audit before submitting and writes model, prompt-template, command, data-manifest, code-hash, and
Slurm job-ID provenance to `runs/`.

After every full-grid job completes, one aggregation pass writes the archived JSON
result and tables into the canonical paper tree:

```bash
.runtime/oat_conda/bin/python exp3_training_transfer/mechanism_family/make_paper_outputs.py \
  --greedy-latex paper/tables/exp3c_structural_ood_greedy.tex \
  --stochastic-latex paper/tables/exp3c_structural_ood_stochastic.tex \
  --overall-latex paper/tables/exp3c_structural_ood_overall.tex \
  --secondary-latex paper/tables/exp3c_structural_ood_secondary.tex
```

The generator requires the exact registered 42-training/6-base roster rather than
accepting job counts alone. Repeated decodes are averaged within episode before the
registered seed-first hierarchical bootstrap. The result manifest hashes every
score file plus the reporting, contrast, and rendering code.

The paper sources use conditional table inputs: they remain claim-free before the
grid finishes and include these generated tables after the verified aggregation.


## Canary gates

Full training remains locked until the six cross-model/cross-disclosure causal canaries have saved a
final adapter and endpoint evaluations, show non-collapsed reward variation, keep the maximum
per-round training truncation rate at or below 5%, and meet at least 95% overall strict JSON parse
rate under both greedy and repeated stochastic decoding. The same fail-closed audit requires all
six untrained-base jobs to complete the exact four-block by three-horizon held-out grid, producing
720 greedy and 3,600 stochastic scored draws per model/disclosure cell with at least 95% parse
coverage for each decode. Recorded scheduler replacements are resolved explicitly; unregistered
substitutions fail. Parse failures remain failed forecasts in
the scientific analysis; the gate detects catastrophic protocol failure rather than selecting on
perfect model behavior. The Polymarket Qwen3-8B/Llama-8B expansion remains downstream of these
synthetic gates; its corpus, split, reward, and sealed test are not changed here.

## Latent-mechanism extension

The completed base and LoRA endpoints now have a frozen, appendix-only
hidden-state study under [`mechanistic_probe/`](mechanistic_probe/README.md).
It probes within-world episode gain and truth-minus-prior impulse responses,
rather than the surface-confounded held-out world label, and compares base,
causal-family, and population-prior endpoints on paired disclosed/undisclosed
prompts. The seed-42 Qwen3-8B grid is a pilot; no training-mechanism claim is
made until seeds 43/44 and the cross-model roster replicate it.
