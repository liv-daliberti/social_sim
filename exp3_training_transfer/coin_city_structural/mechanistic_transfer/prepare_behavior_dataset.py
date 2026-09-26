#!/usr/bin/env python3
"""Convert the frozen task JSONL to the exact endpoint-evaluator dataset contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, STUDY, atomic_json, file_sha256, load_frozen_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "behavior")
    args = parser.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk

    tasks, manifest = load_frozen_tasks()
    records = [
        {
            "input": row["prompt"],
            "reference": json.dumps(row, sort_keys=True),
        }
        for row in tasks
    ]
    if args.output.exists():
        observed = load_from_disk(str(args.output))["train"]
        if len(observed) != len(records):
            raise RuntimeError("existing behavior dataset has wrong length")
    else:
        DatasetDict({"train": Dataset.from_list(records)}).save_to_disk(str(args.output))
    receipt = {
        "study": STUDY,
        "status": "behavior_dataset_ready",
        "path": str(args.output),
        "record_count": len(records),
        "tasks_sha256": manifest["tasks_sha256"],
        "dataset_files": {
            str(path.relative_to(args.output)): file_sha256(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file()
        },
    }
    atomic_json(DATA_DIR / "behavior_dataset_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
