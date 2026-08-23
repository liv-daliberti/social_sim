#!/usr/bin/env python3
"""Remove records for markets not in diverse_2026-06-09.jsonl from a forecast file.

Usage:
    python agent/filter_to_june09_markets.py --model qwen2.5:14b
    python agent/filter_to_june09_markets.py --model qwen2.5:14b --dry-run
"""
import argparse
import json
import re
import shutil
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent
_FC_DIR = _ROOT / "data" / "initial_forecasts"
_SEL    = _ROOT / "data" / "selected_markets"
_REF    = _SEL / "diverse_2026-06-09.jsonl"


def _norm(tid: str) -> str:
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slug = args.model.replace(":", "-").replace("/", "-")

    # Reference: base IDs that belong in the June-09 set
    with _REF.open() as f:
        keep_ids = {_norm(json.loads(l)["task_id"]) for l in f}
    print(f"Reference markets (June-09): {len(keep_ids)}")

    # Find latest forecast file
    candidates = sorted(_FC_DIR.glob(f"forecasts_{slug}_*.jsonl"), reverse=True)
    candidates = [c for c in candidates if "old" not in c.name]
    if not candidates:
        print(f"No forecast file found for {slug}")
        return
    path = candidates[0]
    print(f"Forecast file: {path}")

    kept = removed = 0
    lines_to_keep = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                tid = _norm(rec.get("task_id", ""))
                if tid in keep_ids:
                    lines_to_keep.append(line)
                    kept += 1
                else:
                    removed += 1
            except json.JSONDecodeError:
                lines_to_keep.append(line)

    print(f"Records to keep:   {kept}")
    print(f"Records to remove: {removed}  (June-11-only markets)")

    if args.dry_run:
        print("[dry-run] no changes written")
        return

    # Back up then overwrite
    backup = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, backup)
    print(f"Backup: {backup}")

    with path.open("w") as f:
        for line in lines_to_keep:
            f.write(line + "\n")

    unique = len({_norm(json.loads(l).get("task_id","")) for l in lines_to_keep})
    print(f"Done — {kept} records ({unique} unique markets) written to {path.name}")


if __name__ == "__main__":
    main()
