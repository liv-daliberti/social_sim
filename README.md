# social_sim

Code, frozen data, and manuscript sources for the forecasting and inductive
reasoning paper. Active research code, deployed support services, and historical
work are deliberately separated.

## Active repository map

| Path | Role | Status |
|---|---|---|
| `exp1_prospective/agent/`, `fetch_markets/` | Prospective evidence interventions and frozen analysis | Maintained paper code |
| `exp1_prospective/stage3_materials_annotation/` | Human materials review and deployed review site | Maintained paper code/service |
| `exp2_simulated_worlds/biased_news/` | Appendix coin probe plus the deployed legacy-results viewer | Maintained paper artifact/service |
| `exp2_v2/biased_news/` | Final six-model Coin City response-regime experiment | Maintained |
| `exp3_training_transfer/coin_city_structural/` | Coin City transfer campaign | Complete; exact roster validated |
| `exp3_training_transfer/mechanism_family/` | Structural-OOD campaign | Complete; exact roster validated |
| `exp3_training_transfer/polymarket/` | Historical-market training | Complete three-model locked-test roster |
| `exp3_training_transfer/biased_news/` | Shared RL runtime and frozen news-response world | Maintained internal dependency |
| `exp3_training_transfer/elections/` | Deployed historical Experiment 3 viewer | Maintained service |
| `paper/` | Canonical ICLR manuscript, generators, figures, and tables | Maintained |
| `viewer/`, `exp1_prospective/viewer/` | Deployed cross-experiment and Exp1 viewers | Maintained services |
| `hf_release/` | Anonymous data-release builder and validator | Maintained release tooling |

Every experiment and manuscript archive has a local README. Nothing under an
`_archive/` path is imported, tested, deployed, or packaged by maintained code.
The archives preserve scientific provenance without presenting old entrypoints
as supported workflows. `tests/test_repository_boundaries.py` enforces that
separation for imports, operational entrypoints, and manuscript includes.

## Verification

The repository has one review interface:

```bash
make review
```

This runs static checks and all maintained tests. `make test` and `make lint`
run either half independently; `make help` lists paper and submission commands.

Validate the frozen Experiment 2 design without making an API call:

```bash
cd exp2_v2/biased_news
python eval/validate_coin_city_stable_relationship_claude_n250.py
```

Build the ICLR paper with:

```bash
make paper
```

The Experiment 3 scheduler and paper handoff are tracked in
`exp3_training_transfer/EXPERIMENT_STATUS_PRIORITY.md`. Do not rename shared
Experiment 3 scripts or clear `.runtime/` while jobs are running.

`REPRODUCIBILITY.md` maps each reported claim to its frozen authority, inference
unit, and fail-closed validator.

`paper/ICLR` is a temporary compatibility symlink for already-submitted Slurm
finalizers. It resolves to `paper/`; there is only one physical manuscript tree.

## Artifact policy

Frozen task files, scoring keys, response records, result tables, and files
explicitly cited by the manuscript are research artifacts and remain in place.
Interpreter caches, LaTeX auxiliaries, editor backups, transient runtime trees,
and scheduler logs are ignored. Historical code is archived only when the
current paper and its validators no longer import it.

## GitHub publication boundary

The code repository intentionally excludes raw upstream corpora, rolling market
snapshots, row-level Experiment 1 request/response dumps, generated Arrow
datasets, model artifacts, and archived data payloads. These files remain
available in the research workspace, but they are not suitable Git objects: the
three upstream Polymarket snapshots alone exceed 100 GB. Manifests, checksums,
the frozen June 9 market selection, frozen aggregate results, task builders,
validators, paper tables, and maintained source remain tracked. The anonymous
dataset built by `hf_release/` is the distribution path for the corresponding
public data.

`tests/test_repository_boundaries.py` audits the complete Git candidate set and
fails if a prohibited payload or a file approaching GitHub's 100 MB hard limit
becomes eligible for staging.
