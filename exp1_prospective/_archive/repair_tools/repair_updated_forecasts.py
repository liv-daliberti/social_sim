#!/usr/bin/env python3
"""Recover records from update files damaged by interleaved concurrent writes.

Some update shards were appended to by several workers at once, so one writer's
bytes are spliced into another's mid-line. The damage is byte level: a spliced
record has lost bytes to its neighbour and cannot be reconstructed. What *can*
be recovered are the complete JSON objects that survive intact inside a damaged
line, which the line-at-a-time reader in evaluate_consistency.py discards along
with the fragment it is glued to.

The repair is conservative:

  * candidate records are cut at the '{"update_id"' boundary marker;
  * every complete JSON object in a candidate is pulled out with raw_decode;
  * anything left over is written to a .unrecoverable sidecar, never guessed at;
  * the original file is preserved as .corrupt.bak before rewriting.

No field is edited, inferred, or defaulted. Run with --dry-run first.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

MARKER = '{"update_id"'
UID = re.compile(r'^\{"update_id":\s*"([^"]+)"')


def candidates(text: str) -> list[str]:
    groups: list[list[str]] = []
    current: list[str] | None = None
    for segment in text.split("\n"):
        if segment.startswith(MARKER):
            if current is not None:
                groups.append(current)
            current = [segment]
        elif current is not None:
            current.append(segment)
        elif segment.strip():
            groups.append([segment])
    if current is not None:
        groups.append(current)
    return ["\n".join(g).strip() for g in groups]


def recover(text: str) -> tuple[list[dict], list[str]]:
    decoder = json.JSONDecoder()
    records: list[dict] = []
    orphans: list[str] = []
    for candidate in candidates(text):
        pos = 0
        while pos < len(candidate):
            try:
                obj, end = decoder.raw_decode(candidate, pos)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                records.append(obj)
            pos = end
            while pos < len(candidate) and candidate[pos] in " \n\r\t":
                pos += 1
        if pos < len(candidate):
            orphans.append(candidate[pos:])
    return records, orphans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in args.paths:
        text = path.read_text(encoding="utf-8", errors="surrogateescape")

        baseline: dict[str, dict] = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("update_id"):
                baseline[obj["update_id"]] = obj

        records, orphans = recover(text)
        repaired: dict[str, dict] = {}
        for obj in records:
            uid = obj.get("update_id")
            if not uid:
                continue
            prior = repaired.get(uid)
            if prior is None or (prior.get("delta_yes_prob") is None
                                 and obj.get("delta_yes_prob") is not None):
                repaired[uid] = obj

        gained = set(repaired) - set(baseline)
        dropped = set(baseline) - set(repaired)
        print(f"{path.name}")
        print(f"  records readable today        : {len(baseline)}")
        print(f"  records after repair          : {len(repaired)}")
        print(f"  net new records recovered     : {len(gained)}"
              f" ({sum(1 for u in gained if repaired[u].get('delta_yes_prob') is not None)}"
              f" with a usable delta)")
        print(f"  records lost by repair        : {len(dropped)}")
        print(f"  unrecoverable fragments       : {len(orphans)}")
        if dropped:
            raise SystemExit("refusing to rewrite: repair would drop records")
        if args.dry_run:
            continue

        backup = path.with_suffix(path.suffix + ".corrupt.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        if orphans:
            path.with_suffix(path.suffix + ".unrecoverable").write_text(
                "\n".join(orphans), encoding="utf-8", errors="surrogateescape")
        with path.open("w", encoding="utf-8") as handle:
            for uid in sorted(repaired):
                handle.write(json.dumps(repaired[uid], ensure_ascii=False) + "\n")
        print(f"  rewrote {path.name}; original kept at {backup.name}")


if __name__ == "__main__":
    main()
