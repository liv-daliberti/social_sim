#!/usr/bin/env python3
"""Recover answers whose JSON key carries stray whitespace.

The frozen reply parser requires the exact key ``predicted_poll``. A model that
emits ``{"rationale": "...", " predicted_poll": 59.2}`` therefore records as
unparseable even though the reply is valid JSON and states an answer. This pass
re-reads the stored ``raw`` text of records already marked unparseable and fills
in the answer when a whitespace-stripped key matches.

It issues no API calls, so no task receives a second model attempt; it only
changes how an already-recorded reply is read. The rule is applied uniformly to
every model and arm, and every touched record keeps its original ``raw`` text and
is flagged ``reparsed_whitespace_key``.

    python eval/reparse_whitespace_keys.py --apply
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


RESPONSES = ROOT / "data" / EXPERIMENT / "responses"


def recover(raw: str):
    """Return a poll in [0,100] if a whitespace-stripped key supplies one."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        normalized = {str(k).strip(): v for k, v in value.items()}
        if "predicted_poll" not in normalized:
            continue
        try:
            poll = float(normalized["predicted_poll"])
        except (TypeError, ValueError):
            continue
        if 0.0 <= poll <= 100.0:
            return poll, str(normalized.get("rationale", ""))[:500]
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    recovered, inspected = [], 0
    for path in sorted(RESPONSES.glob("responses_*.jsonl")):
        if "corrupt" in path.name:
            continue
        rows, changed = [], False
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("predicted_poll") is None and (row.get("raw") or "").strip():
                inspected += 1
                poll, rationale = recover(row["raw"])
                if poll is not None:
                    row["predicted_poll"] = poll
                    if rationale:
                        row["rationale"] = rationale
                    row["error"] = None
                    row["reparsed_whitespace_key"] = True
                    changed = True
                    recovered.append(
                        {
                            "model": row.get("model"),
                            "arm": row.get("arm"),
                            "task_id": row.get("task_id"),
                            "predicted_poll": poll,
                        }
                    )
            rows.append(row)
        if changed and args.apply:
            path.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
            )

    summary = {
        "applied": args.apply,
        "rule": "strip whitespace from JSON keys before requiring predicted_poll",
        "api_calls_issued": 0,
        "unparseable_records_inspected": inspected,
        "records_recovered": len(recovered),
        "recovered": recovered,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.apply and recovered:
        audit = RESPONSES / "reparse_whitespace_keys.json"
        audit.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
