# Experiment 1 archive

This directory contains Experiment 1 material outside the current paper path:

- `unused_paper_directions/` preserves the unreported indirect-intervention
  study and its planning documents.
- `legacy_pipeline_scripts/` preserves finished daily/Slurm launchers.
- `repair_tools/` preserves one-time data-recovery utilities.
- `interim_results/` preserves pre-authority June report snapshots.
- `legacy_pipeline_config/` preserves the unused status helper and declarative
  local-model config (the maintained runners use CLI defaults and never read it).

Maintained code and frozen analysis artifacts remain outside `_archive/`; no
active test, paper generator, deployment, or runtime entrypoint imports this tree.
