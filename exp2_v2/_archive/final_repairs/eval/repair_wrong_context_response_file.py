#!/usr/bin/env python3
"""Repair a wrong-context response file damaged by concurrent appends.

Five shards append to one per-model file. On this network filesystem that is not
reliably atomic for records larger than a page, and one run left NUL padding
interleaved into the Kimi K3 file together with duplicate task records written
while the file was mid-write.

This pass strips the NUL padding, keeps the earliest surviving record per task,
verifies every kept record against the frozen task file's prompt hash, and
reports the tasks that no longer have one. It never invents a record: tasks whose
only copies were destroyed come out as missing and must be re-issued.

    python eval/repair_wrong_context_response_file.py --model FW-Kimi-K3 [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import EXPERIMENT  # noqa: E402
from engine.coin_city_wrong_context_arm import ARM  # noqa: E402


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    expected = {
        json.loads(line)["task_id"]: json.loads(line)["prompt_sha256"]
        for line in (DESIGN / f"tasks_{ARM}.jsonl").read_text().splitlines()
        if line.strip()
    }
    path = RESPONSES / f"responses_{args.model}_{ARM}.jsonl"
    raw = path.read_bytes()

    kept: dict = {}
    unrecoverable = 0
    duplicates = 0
    hash_mismatch = 0
    for chunk in raw.split(b"\n"):
        text = chunk.replace(b"\x00", b"").decode("utf-8", "replace").strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            unrecoverable += 1
            continue
        task_id = record.get("task_id")
        if task_id not in expected:
            unrecoverable += 1
            continue
        if record.get("prompt_sha256") != expected[task_id]:
            hash_mismatch += 1
            continue
        previous = kept.get(task_id)
        if previous is None:
            kept[task_id] = record
        else:
            duplicates += 1
            if record.get("created_at", "") < previous.get("created_at", ""):
                kept[task_id] = record

    missing = sorted(set(expected) - set(kept))
    summary = {
        "model": args.model,
        "arm": ARM,
        "applied": args.apply,
        "bytes_of_nul_padding": raw.count(0),
        "records_kept": len(kept),
        "duplicate_records_dropped": duplicates,
        "unrecoverable_lines": unrecoverable,
        "prompt_hash_mismatches_dropped": hash_mismatch,
        "missing_task_ids": missing,
        "missing_count": len(missing),
        "parsed_kept": sum(r.get("predicted_poll") is not None for r in kept.values()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.apply:
        ordered = [kept[task_id] for task_id in sorted(kept)]
        temporary = path.with_suffix(".jsonl.repaired")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
        )
        temporary.replace(path)
        audit = RESPONSES / f"responses_{args.model}_{ARM}.repair.json"
        audit.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
