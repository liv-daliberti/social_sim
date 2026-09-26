---
pretty_name: "Inductive Forecasting Study — Anonymous Data Release"
language:
- en
license: other
task_categories:
- text-generation
- reinforcement-learning
tags:
- forecasting
- reasoning
- prediction-markets
- synthetic-data
- behavioral-evaluation
configs:
- config_name: exp1_updates
  data_files:
  - split: test
    path: data/exp1/updates.parquet
- config_name: exp1_review_materials
  data_files:
  - split: test
    path: data/exp1/review_materials.parquet
- config_name: exp2_coin_city_tasks
  data_files:
  - split: test
    path: data/exp2/tasks.parquet
- config_name: exp2_coin_city_responses
  data_files:
  - split: test
    path: data/exp2/responses.parquet
- config_name: exp3_coin_city_transfer
  data_files:
  - split: train_causal
    path: data/exp3_coin_city/train_causal.parquet
  - split: train_population_prior
    path: data/exp3_coin_city/train_population_prior.parquet
  - split: train_structureless
    path: data/exp3_coin_city/train_structureless.parquet
  - split: test
    path: data/exp3_coin_city/test.parquet
- config_name: exp3_coin_city_scores
  data_files:
  - split: test
    path: data/exp3_coin_city/scores.parquet
- config_name: exp4_historical_markets
  data_files:
  - split: train
    path: data/exp4/train.parquet
  - split: validation
    path: data/exp4/validation.parquet
  - split: test
    path: data/exp4/test.parquet
- config_name: exp4_locked_test_outputs
  data_files:
  - split: test
    path: data/exp4/locked_test_outputs.parquet
- config_name: exp4_evidence_updates
  data_files:
  - split: test
    path: data/exp4/evidence_updates.parquet
- config_name: appendix_mechanism_transfer
  data_files:
  - split: train
    path: data/appendix_mechanism/train_*.parquet
  - split: test
    path: data/appendix_mechanism/test_*.parquet
- config_name: appendix_mechanism_scores
  data_files:
  - split: test
    path: data/appendix_mechanism/scores.parquet
---

# Inductive Forecasting Study — Anonymous Data Release

This repository is the anonymous data companion to a paper studying behavioral
signatures of inductive reasoning in language-model forecasts. It packages the
frozen inputs, model responses, row-level scores, and aggregate result artifacts
used by the paper's four reported experiments.

The release is organized as Hugging Face dataset configurations so each study can
be loaded independently:

```python
from datasets import load_dataset

updates = load_dataset("ANONYMOUS_NAMESPACE/ANONYMOUS_DATASET", "exp1_updates")
coin_city = load_dataset(
    "ANONYMOUS_NAMESPACE/ANONYMOUS_DATASET",
    "exp2_coin_city_responses",
)
historical = load_dataset(
    "ANONYMOUS_NAMESPACE/ANONYMOUS_DATASET",
    "exp4_historical_markets",
)
```

## Contents

### Experiment 1: selective updating

`exp1_updates` contains one row per frozen model × market × evidence packet ×
run record from the June 17 numerical authority. It includes initial and updated
probabilities, evidence direction, EHC/HFC/ICS indicators, and parsing status.
The source report and threshold analysis are under `artifacts/exp1/`.

`exp1_review_materials` contains the 18 blinded fictional reports shown in the
human materials review. Individual participant ratings and timestamps are not
released because participation consent did not explicitly establish consent for
public row-level redistribution. Aggregate review results remain available under
`artifacts/exp1/`.

### Experiment 2: Coin City

`exp2_coin_city_tasks` contains the frozen target-only, no-context,
correct-context, misleading-context, and arbitrary-symbol prompts joined to their
answer key. `exp2_coin_city_responses` contains all production model responses,
including the matched arbitrary-symbol control set, joined to the same task and
gold fields. Failed pilots and corrupt pre-repair files are excluded.

### Experiment 3: trained Coin City transfer

`exp3_coin_city_transfer` contains the 4,800-row causal, population-prior, and
structureless training sets plus the common 1,440-row held-out evaluation set.
`exp3_coin_city_scores` contains the registered greedy and five-draw Qwen3-4B
endpoint score rows for the base, causal, population-prior, and structureless
conditions. Checkpoints are not part of this dataset repository.

### Experiment 4: historical-market transfer

`exp4_historical_markets` contains the checksum-locked 1,736/512/1,024
train/development/test tasks derived from a public Polymarket archive.
`exp4_locked_test_outputs` contains base and three-seed final forecasts.
`exp4_evidence_updates` contains the secondary paired evidence-update outputs.
The 21 GB upstream scrape is not redistributed.

### Structural-OOD mechanism study

The remaining configurations contain the registered mechanism-composition
train/evaluation datasets and the exact score ledger used by the paper.

## Provenance and construction

- Real-market questions and historical prices were collected from Polymarket's
  public interfaces. The derived task rows preserve the study's point-in-time
  filtering and family-disjoint split.
- Coin City and mechanism-family tasks are synthetic and generated
  by the study code.
- Model response rows are machine-generated outputs from the deployments named in
  each record or accompanying artifact.
- `release_manifest.json` records row counts and explicit exclusions.
- `SHA256SUMS` authenticates every published file.

Absolute cluster paths and author-identifying endpoint names are not included.
Authentication secrets were never intentionally written to the source records;
the release builder also performs a fail-closed secret and identity scan.

## Limitations and responsible use

The synthetic studies are behavioral benchmarks, not demonstrations of a
particular internal representation. The historical-market tasks inherit selection
biases and limitations of prediction-market archives. Static question/rules text
cannot always be proven point-in-time, as described in the paper. Model outputs
may contain errors or unsupported claims and should not be treated as factual or
as financial advice.

Some Experiment 1 records contain fictional, explicitly labeled evidence packets
about real people and events. Preserve the fictional notice when displaying or
redistributing these materials.

## Licensing

See `LICENSE.md`. The repository uses `license: other` because it combines
original synthetic benchmarks with third-party market-derived fields and model
outputs that are subject to their respective source/provider terms. No blanket
relicensing of third-party material is asserted.

## Citation

During double-blind review, cite this repository as:

> Anonymous Authors (2026). *Inductive Forecasting Study — Anonymous Data
> Release*. Hugging Face Datasets.

The card and citation should be updated after deanonymization.
