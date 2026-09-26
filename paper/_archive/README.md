# Paper archive

Nothing under this directory is an input to the maintained ICLR manuscript in
`paper/`.

- `legacy_exp2/` contains superseded three-city and pilot Coin City figures,
  tables, and generators.
- `manuscript_snapshots/` contains older manuscript layouts and forecasting
  drafts retained for historical reference.
- `unused_wrappers/` contains figure wrappers and superseded figure variants
  that are not included by either current manuscript.
- `unused_current_tree_second_pass/` contains wrappers, tables, and figure
  outputs proven absent from both current LaTeX recorder graphs.
- `tooling/` contains retired one-off visual checks.
- `iclr_backup/` contains inactive files from an older ICLR working copy.
  The bibliography and style files still required by current builds live in
  `paper/`.
- `editor_backups/` and `exports/` are recovery-only copies and exports.
- `retired_tables/` contains superseded result presentations that are retained
  for provenance but excluded from the maintained manuscript.
- `retired_iclr_mirror_*/` is a local, ignored recovery copy created when the
  duplicate submission tree was consolidated into `paper/`. Git history retains
  the former tracked tree; the local copy can be removed after queued Slurm
  finalizers have completed.

Restore a file to the active tree only after confirming that a maintained
manuscript or generation script actually depends on it.
