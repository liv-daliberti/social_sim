#!/usr/bin/env python3
"""Join private labels offline and apply the pilot's fail-closed human gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DB = ROOT / "data/annotations/pilot_v2_reviews.sqlite3"
DEFAULT_CANDIDATES = ROOT / "data/development/pilot_candidates.jsonl"
DEFAULT_REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
PROTOCOL_VERSION = "indirect-pilot-v2"
LABEL_TO_RESPONSE = {
    "increase_yes": "increase",
    "decrease_yes": "decrease",
    "no_material_effect": "no_effect",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_id(panel: str, candidate_id: str, role: str) -> str:
    raw = f"{PROTOCOL_VERSION}|{panel}|{candidate_id}|{role}".encode()
    return "review_" + hashlib.sha256(raw).hexdigest()[:18]


def med(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"INCOMPLETE: database not found: {args.db}")

    candidates = read_jsonl(args.candidates)
    reviewer_file = json.loads(args.reviewers.read_text(encoding="utf-8"))
    expected_annotations = sum(len(row["task_ids"]) for row in reviewer_file["reviewers"])

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    annotations = [dict(row) for row in connection.execute("SELECT * FROM annotations").fetchall()]
    connection.close()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_task[row["task_id"]].append(row)

    task_results = []
    all_full_pass = True
    all_shortcut_pass = True
    for candidate in candidates:
        for role, context in candidate["contexts"].items():
            task_id = opaque_id("full_context", candidate["candidate_id"], role)
            rows = by_task.get(task_id, [])
            expected = LABEL_TO_RESPONSE[context["intended_label"]]
            directions = Counter(row["direction"] for row in rows)
            complete = len(rows) == 3
            direction_pass = complete and directions == Counter({expected: 3})
            confidence_pass = complete and med(rows, "confidence") >= 4
            passed = direction_pass and confidence_pass
            all_full_pass = all_full_pass and passed
            task_results.append({
                "candidate_id": candidate["candidate_id"],
                "role": role,
                "task_id": task_id,
                "ratings": len(rows),
                "directions": dict(directions),
                "expected_direction": expected,
                "median_confidence": med(rows, "confidence") if rows else None,
                "passes_human_gate": passed,
            })

        task_id = opaque_id("shortcut", candidate["candidate_id"], "masked")
        rows = by_task.get(task_id, [])
        directions = Counter(row["direction"] for row in rows)
        complete = len(rows) == 3
        high_confidence_fixed_sign = (
            complete
            and len(directions) == 1
            and next(iter(directions)) in {"increase", "decrease"}
            and med(rows, "confidence") >= 4
        )
        passed = complete and not high_confidence_fixed_sign
        all_shortcut_pass = all_shortcut_pass and passed
        task_results.append({
            "candidate_id": candidate["candidate_id"],
            "role": "shortcut_masked",
            "task_id": task_id,
            "ratings": len(rows),
            "directions": dict(directions),
            "high_confidence_fixed_sign": high_confidence_fixed_sign,
            "passes_shortcut_gate": passed,
        })

    complete = len(annotations) == expected_annotations and all(
        len(rows) == 3 for rows in by_task.values()
    ) and len(by_task) == 12
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_count": len(annotations),
        "expected_annotation_count": expected_annotations,
        "complete": complete,
        "full_context_gate": complete and all_full_pass,
        "shortcut_gate": complete and all_shortcut_pass,
        "pilot_ready_for_revision": complete and all_full_pass and all_shortcut_pass,
        "task_results": task_results,
        "interpretation": (
            "A passing pilot validates these draft items and the human workflow only. "
            "It does not validate the frozen core or authorize target-model calls."
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not summary["pilot_ready_for_revision"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
