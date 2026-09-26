# Retired Experiment 2 work

This directory contains experiment iterations that the current manuscript no
longer reports or executes.

- `three_city/` contains the structure-blind v1-v13 design sequence, including
  preregistrations, source, tests, logs, and frozen outputs.
- `coin_city_pilots/` contains intermediate Coin City generators, renderers,
  and data roots superseded by
  `coin_city_stable_relationship_claude_n250_v4`.
- `final_repairs/` contains one-time response compaction and repair utilities;
  the corrected frozen responses remain in the maintained data tree.

These files are retained for scientific provenance, not as supported entry
points. Their original relative layout is represented inside each archive, but
imports and launch scripts may rely on the former working-tree location. The
maintained workflow is documented in `../README.md` and must not import from
this archive.
