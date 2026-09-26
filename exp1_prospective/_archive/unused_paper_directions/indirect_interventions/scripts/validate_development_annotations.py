#!/usr/bin/env python3
"""Validate the 20-market development panel before any core generation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from validate_core_annotations import (
    fleiss_kappa,
    load_annotations,
    read_jsonl,
    task_gate,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_ANNOTATIONS = ROOT / "data/annotations/development_reviews.sqlite3"
DEFAULT_REVIEWERS = ROOT / "data/review/development_reviewers.json"
DEFAULT_KEY = ROOT / "data/review/development_review_key.jsonl"
DEFAULT_CANDIDATES = ROOT / "data/development/development_candidates_readable_v3.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/development_gate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.annotations, args.reviewers, args.private_key, args.candidates):
        if not path.exists():
            raise SystemExit(f"INCOMPLETE: {path}")
    candidates = read_jsonl(args.candidates)
    if len(candidates) != 60 or len({row["market"]["task_id"] for row in candidates}) != 20:
        raise SystemExit("development pool must be exactly 20 markets x 3 families")
    editor_gate = all(
        row.get("generation", {}).get("status") == "human_edited"
        and row["generation"].get("editor_id")
        and row["generation"].get("edited_at")
        for row in candidates
    )
    reviewers = json.loads(args.reviewers.read_text(encoding="utf-8"))
    ratings_per_task = int(reviewers.get("ratings_per_task", 3))
    expected_pairs = {
        (reviewer["reviewer_id"], task_id)
        for reviewer in reviewers["reviewers"]
        for task_id in reviewer["task_ids"]
    }
    annotations = load_annotations(args.annotations)
    actual_pairs = {(row["reviewer_id"], row["task_id"]) for row in annotations}
    coverage = len(actual_pairs) == len(annotations) and actual_pairs == expected_pairs
    by_task = defaultdict(list)
    for row in annotations:
        by_task[row["task_id"]].append(row)
    key_rows = read_jsonl(args.private_key)
    roles = defaultdict(dict)
    shortcut = {}
    full_groups = []
    for key in key_rows:
        rows = by_task.get(key["task_id"], [])
        if key["panel"] == "shortcut":
            directions = {row["direction"] for row in rows}
            confidence = sorted(float(row["confidence"]) for row in rows)
            median = confidence[len(confidence) // 2] if confidence else None
            fixed = (
                len(rows) == ratings_per_task
                and len(directions) == 1
                and next(iter(directions)) in {"increase", "decrease"}
                and median >= 4
            )
            shortcut[key["source_id"]] = len(rows) == ratings_per_task and not fixed
        else:
            passed, _ = task_gate(rows, key["expected_label"], ratings_per_task)
            roles[key["source_id"]][key["role"]] = passed
            if len(rows) == ratings_per_task:
                full_groups.append([row["direction"] for row in rows])
    passed_families = [
        source_id for source_id, values in roles.items()
        if values == {"positive": True, "negative": True, "broken": True}
        and shortcut.get(source_id, False)
    ]
    kappa = fleiss_kappa(full_groups)
    # Development is a prompt-quality gate, not a paper endpoint. Requiring at
    # least 80% clean families exposes systematic design failures before core use.
    family_gate = len(passed_families) >= 48
    agreement_gate = kappa is not None and kappa >= 0.70
    ready = coverage and editor_gate and family_gate and agreement_gate
    report = {
        "protocol_version": "indirect-development-v1",
        "ready_for_core_generation": ready,
        "coverage_pass": coverage,
        "human_editing_pass": editor_gate,
        "family_count": len(candidates),
        "passing_family_count": len(passed_families),
        "minimum_passing_families": 48,
        "family_gate": family_gate,
        "fleiss_kappa_direction": kappa,
        "agreement_gate": agreement_gate,
        "passing_candidate_ids": sorted(passed_families),
        "instruction": (
            "freeze the generation prompt and proceed to core candidates"
            if ready else "revise using development annotations only and recollect fresh ratings"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
