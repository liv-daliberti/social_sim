#!/usr/bin/env python3
"""Load the dataset exactly as the trainer does, before any job is submitted.

This exists because `validate_dataset.py` passed all 28 of its checks against a
raw `.jsonl` while the trainer requires a HuggingFace Arrow directory. Six jobs
went to the cluster, every one died with FileNotFoundError about 80 seconds in,
and then hung holding two A6000s each until they were killed 7.5 hours later.
Validating the content of an artifact says nothing about whether the thing that
consumes it can open it.

So this reproduces the real call chain:

    train.sh  --prompt_data $DATA/train --train_split train
              --input_key input --output_key reference
    oat.learners.base.prepare_data
      -> oat.utils.data.load_data_from_disk_or_hf(prompt_data)[train_split]
        -> datasets.load_from_disk(...)

Must be run with the oat interpreter, which is the one the jobs use:
    /n/fs/similarity/social_sim/.runtime/oat_conda/bin/python
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ARMS = ("causal", "population_prior")
SPLITS = {"train": 4800, "heldout": 2880}
TRAIN_SPLIT = "train"
INPUT_KEY = "input"
OUTPUT_KEY = "reference"


def main() -> None:
    try:
        from oat.utils.data import load_data_from_disk_or_hf
    except ImportError as exc:
        raise SystemExit(
            f"cannot import oat ({exc}). Run this with the oat interpreter:\n"
            "  /n/fs/similarity/social_sim/.runtime/oat_conda/bin/python "
            f"{Path(__file__).name}")

    failures = []
    for arm in ARMS:
        for split, expected in SPLITS.items():
            path = DATA / arm / split
            label = f"{arm}/{split}"
            try:
                loaded = load_data_from_disk_or_hf(str(path))
            except Exception as exc:                       # noqa: BLE001
                failures.append(f"{label}: trainer loader raised {type(exc).__name__}: {exc}")
                continue
            if TRAIN_SPLIT not in loaded:
                failures.append(f"{label}: no '{TRAIN_SPLIT}' split, has {list(loaded)}")
                continue
            table = loaded[TRAIN_SPLIT]
            if len(table) != expected:
                failures.append(f"{label}: {len(table)} rows, expected {expected}")
            missing = [k for k in (INPUT_KEY, OUTPUT_KEY) if k not in table.column_names]
            if missing:
                failures.append(f"{label}: missing column(s) {missing}, "
                                f"has {table.column_names}")
                continue
            row = table[0]
            if not isinstance(row[INPUT_KEY], str) or not row[INPUT_KEY].strip():
                failures.append(f"{label}: '{INPUT_KEY}' is not a non-empty string")
            if not isinstance(row[OUTPUT_KEY], str) or not row[OUTPUT_KEY].strip():
                failures.append(f"{label}: '{OUTPUT_KEY}' is not a non-empty string")
            if not failures:
                print(f"  ok   {label}: {len(table)} rows, columns {table.column_names}")

    if failures:
        print("\nTRAINER-LOAD PREFLIGHT FAILED:")
        for line in failures:
            print(f"  FAIL {line}")
        sys.exit(1)
    print("\ntrainer-load preflight passed: every split opens through oat's own loader")


if __name__ == "__main__":
    main()
