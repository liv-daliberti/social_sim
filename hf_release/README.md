# Hugging Face data release

This directory builds the anonymous, viewer-friendly dataset repository that
accompanies the paper. The generated repository is written to `build/` and is
ignored by Git.

## Build and validate

From the repository root:

```bash
python hf_release/build_release.py
python hf_release/validate_release.py hf_release/build
```

The builder uses only Python's standard library and `pyarrow`. It converts the
registered Hugging Face Arrow datasets to compressed Parquet, normalizes the
paper's JSON/JSONL results, removes absolute local paths, writes a dataset card,
and produces `release_manifest.json` plus `SHA256SUMS`.

The large row-level inputs are intentionally ignored by the Git code repository.
Run the builder from the complete research workspace, or use the separately
published dataset release when working from a GitHub clone.

## Publish

Publishing requires a neutral Hugging Face organization or account so the URL
does not identify the authors during double-blind review.

For strict double-blind review, prefer a separate neutral user account. Standard
organization pages can expose their member list; Hugging Face documents member-list
hiding as a Team/Enterprise control, and warns that membership can still be found
through other means. A neutral organization name alone is therefore not a complete
anonymity boundary.

```bash
python -m venv --system-site-packages hf_release/.venv
hf_release/.venv/bin/python -m pip install -r hf_release/requirements.txt
hf_release/.venv/bin/hf auth login
hf_release/.venv/bin/python hf_release/build_release.py --force \
  --repo-id neutral-namespace/inductive-forecasting-data
hf_release/.venv/bin/python hf_release/validate_release.py hf_release/build
hf_release/.venv/bin/python hf_release/push_release.py \
  neutral-namespace/inductive-forecasting-data
```

The push helper refuses a personal namespace by default when it can identify the
logged-in user. For a separately created neutral user account, review its public
profile and pass `--allow-personal-namespace`; never use that override for an
author-identifying account. Verify the public profile before placing the resulting
URL in the paper.

## Release boundary

Included:

- Experiment 1's exact frozen row-level analysis, blinded review materials, and
  aggregate reports;
- Experiment 2's frozen Coin City tasks, answer keys, production responses, and
  additive controls;
- Experiments 3 and 4's registered train/evaluation datasets, final score ledgers,
  historical-market tasks, and endpoint outputs;
- the structural-OOD mechanism datasets and the final score ledger named by the
  frozen analysis manifest.

Excluded:

- model checkpoints, caches, debug logs, failed pilots, corrupt backups, and
  mutable post-freeze collection snapshots;
- the 21 GB upstream Polymarket scrape (the checksum-locked derived train/dev/test
  tasks are included instead);
- individual human-review ratings, timestamps, and durations, because consent to
  participate is not the same as consent to public row-level release.
