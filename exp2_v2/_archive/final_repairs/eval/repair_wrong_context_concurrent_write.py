#!/usr/bin/env python3
"""Repair a wrong-context response file damaged by concurrent appends.

A resumed shard was started while stragglers from an interrupted controller were
still writing to the same JSONL, so the file picked up interleaved and
NUL-padded lines and duplicate task IDs. This pass quarantines the damaged file
and rewrites it keeping, for each task, the *earliest* record carrying a model
response --- the first attempt the model actually served. Later duplicates are
discarded rather than averaged, so no task contributes twice and no task is
re-asked to replace a response that already exists.

Tasks left with no valid record are reported; rerunning the arm refills exactly
those, since every surviving record is terminal.

    python eval/repair_wrong_context_concurrent_write.py --model FW-Kimi-K3 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    task_id,
)
from engine.coin_city_wrong_context_arm import ARM  # noqa: E402


RUN = ROOT / "data" / EXPERIMENT
RESPONSES = RUN / "responses"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    path = RESPONSES / f"responses_{args.model}_{ARM}.jsonl"
    rows, malformed = [], 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed += 1

    keep: dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        identifier = row.get("task_id")
        if not identifier:
            continue
        current = keep.get(identifier)
        if current is None:
            keep[identifier] = row
            continue
        duplicates += 1
        # Prefer the earliest record that carries a model response.
        current_ok = current.get("response_received") is True
        row_ok = row.get("response_received") is True
        if (row_ok and not current_ok) or (
            row_ok == current_ok
            and row.get("created_at", "") < current.get("created_at", "")
        ):
            keep[identifier] = row

    expected = [
        task_id(episode, k) for episode in range(EPISODES) for k in C_CASE_LEVELS
    ]
    missing = [identifier for identifier in expected if identifier not in keep]
    summary = {
        "model": args.model,
        "arm": ARM,
        "applied": args.apply,
        "lines_parsed": len(rows),
        "malformed_lines_discarded": malformed,
        "duplicate_records_discarded": duplicates,
        "unique_tasks_kept": len(keep),
        "tasks_missing_after_repair": len(missing),
        "missing_task_ids": missing,
        "reason": (
            "concurrent appends by a resumed shard and an interrupted "
            "controller's stragglers"
        ),
        "resolution": (
            "kept the earliest record carrying a model response for each task; "
            "no task contributes more than one response"
        ),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_task_ids"},
                     indent=2, sort_keys=True))
    if missing[:10]:
        print("first missing:", missing[:10])

    if args.apply:
        quarantine = RESPONSES / f"responses_{args.model}_{ARM}.corrupt.jsonl"
        shutil.copy2(path, quarantine)
        ordered = [keep[i] for i in expected if i in keep]
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered)
        )
        audit = RESPONSES / f"responses_{args.model}_{ARM}.repair.json"
        audit.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"quarantined -> {quarantine.name}; rewrote {len(ordered)} records")


if __name__ == "__main__":
    main()
