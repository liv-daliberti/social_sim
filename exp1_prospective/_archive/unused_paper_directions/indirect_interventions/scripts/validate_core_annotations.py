#!/usr/bin/env python3
"""Join blinded core ratings offline and apply all registered retention gates."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_ANNOTATIONS = ROOT / "data/annotations/core_reviews.sqlite3"
DEFAULT_REVIEWERS = ROOT / "data/review/core_reviewers.json"
DEFAULT_KEY = ROOT / "data/review/core_review_key.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/core_human_validation.json"
LABEL_TO_RESPONSE = {
    "increase_yes": "increase",
    "decrease_yes": "decrease",
    "no_material_effect": "no_effect",
}
CATEGORIES = ("increase", "decrease", "no_effect", "ambiguous")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix in {".sqlite", ".sqlite3", ".db"}:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM annotations")]
        connection.close()
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["annotations"] if isinstance(payload, dict) else payload


def fleiss_kappa(groups: list[list[str]]) -> float | None:
    if not groups:
        return None
    n = len(groups[0])
    if n < 2 or any(len(group) != n for group in groups):
        return None
    counts = [Counter(group) for group in groups]
    p_bar = sum(
        (sum(counts_i[category] ** 2 for category in CATEGORIES) - n) / (n * (n - 1))
        for counts_i in counts
    ) / len(counts)
    total = len(groups) * n
    marginals = {
        category: sum(row[category] for row in counts) / total
        for category in CATEGORIES
    }
    p_expected = sum(value * value for value in marginals.values())
    if p_expected == 1:
        return 1.0
    return (p_bar - p_expected) / (1 - p_expected)


def task_gate(
    rows: list[dict[str, Any]],
    expected_label: str,
    ratings_per_task: int,
) -> tuple[bool, dict[str, Any]]:
    expected = LABEL_TO_RESPONSE[expected_label]
    directions = Counter(row["direction"] for row in rows)
    confidence = (
        statistics.median(float(row["confidence"]) for row in rows)
        if rows else None
    )
    complete = (
        len(rows) == ratings_per_task
        and len({row["reviewer_id"] for row in rows}) == ratings_per_task
    )
    passed = complete and directions == Counter({expected: ratings_per_task}) and confidence >= 4
    return passed, {
        "ratings": len(rows),
        "directions": dict(directions),
        "expected": expected,
        "median_confidence": confidence,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--compensation", default="not recorded")
    args = parser.parse_args()
    for path in (args.annotations, args.reviewers, args.private_key):
        if not path.exists():
            raise SystemExit(f"INCOMPLETE: required input not found: {path}")

    annotations = load_annotations(args.annotations)
    reviewer_file = json.loads(args.reviewers.read_text(encoding="utf-8"))
    key_rows = read_jsonl(args.private_key)
    ratings_per_task = int(reviewer_file.get("ratings_per_task", 3))
    expected_pairs = {
        (reviewer["reviewer_id"], task_id)
        for reviewer in reviewer_file["reviewers"]
        for task_id in reviewer["task_ids"]
    }
    actual_pairs = {(row["reviewer_id"], row["task_id"]) for row in annotations}
    exact_coverage = (
        len(actual_pairs) == len(annotations)
        and actual_pairs == expected_pairs
    )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_task[row["task_id"]].append(row)

    task_results: dict[str, dict[str, Any]] = {}
    family_roles: dict[str, dict[str, bool]] = defaultdict(dict)
    family_markets: dict[str, str] = {}
    control_pass: dict[str, bool] = {}
    control_markets: dict[str, str] = {}
    shortcut_pass: dict[str, bool] = {}
    full_groups: list[list[str]] = []
    for key in key_rows:
        rows = by_task.get(key["task_id"], [])
        if key["panel"] == "shortcut":
            directions = Counter(row["direction"] for row in rows)
            confidence = (
                statistics.median(float(row["confidence"]) for row in rows)
                if rows else None
            )
            complete = (
                len(rows) == ratings_per_task
                and len({row["reviewer_id"] for row in rows}) == ratings_per_task
            )
            fixed_sign = (
                complete
                and len(directions) == 1
                and next(iter(directions)) in {"increase", "decrease"}
                and confidence >= 4
            )
            passed = complete and not fixed_sign
            shortcut_pass[key["source_id"]] = passed
            task_results[key["task_id"]] = {
                "source_id": key["source_id"],
                "role": "masked",
                "ratings": len(rows),
                "directions": dict(directions),
                "median_confidence": confidence,
                "high_confidence_fixed_sign": fixed_sign,
                "pass": passed,
            }
            continue

        passed, detail = task_gate(rows, key["expected_label"], ratings_per_task)
        task_results[key["task_id"]] = {
            "source_id": key["source_id"],
            "role": key["role"],
            **detail,
        }
        if len(rows) == ratings_per_task:
            full_groups.append([row["direction"] for row in rows])
        if key["source_type"] == "family":
            family_roles[key["source_id"]][key["role"]] = passed
            family_markets[key["source_id"]] = key["market_task_id"]
        else:
            control_pass[key["source_id"]] = passed
            control_markets[key["source_id"]] = key["market_task_id"]

    retained_families = sorted(
        source_id for source_id, roles in family_roles.items()
        if roles == {"positive": True, "negative": True, "broken": True}
        and shortcut_pass.get(source_id, False)
    )
    families_by_market: dict[str, list[str]] = defaultdict(list)
    for source_id in retained_families:
        families_by_market[family_markets[source_id]].append(source_id)
    controls_by_market: dict[str, list[str]] = defaultdict(list)
    for source_id, passed in control_pass.items():
        if passed:
            controls_by_market[control_markets[source_id]].append(source_id)
    retained_markets = sorted(
        market_id for market_id, family_ids in families_by_market.items()
        if len(family_ids) == 3 and len(controls_by_market.get(market_id, [])) == 4
    )
    retained_market_set = set(retained_markets)
    retained_families = [
        source_id for source_id in retained_families
        if family_markets[source_id] in retained_market_set
    ]
    retained_controls = sorted(
        source_id for source_id, passed in control_pass.items()
        if passed and control_markets[source_id] in retained_market_set
    )
    kappa = fleiss_kappa(full_groups)
    agreement_gate = kappa is not None and kappa >= 0.70
    sample_gate = len(retained_markets) >= 80 and len(retained_families) >= 240
    ready = exact_coverage and agreement_gate and sample_gate
    report = {
        "protocol_version": "indirect-core-v1",
        "round": args.round,
        "ready_to_freeze": ready,
        "coverage": {
            "annotation_count": len(annotations),
            "expected_annotation_count": len(expected_pairs),
            "exact": exact_coverage,
        },
        "raters": {
            "count": len(reviewer_file["reviewers"]),
            "ratings_per_task": ratings_per_task,
            "compensation": args.compensation,
        },
        "agreement": {
            "fleiss_kappa_direction": kappa,
            "minimum": 0.70,
            "pass": agreement_gate,
        },
        "retention": {
            "market_count": len(retained_markets),
            "family_count": len(retained_families),
            "control_count": len(retained_controls),
            "minimum_markets": 80,
            "minimum_families": 240,
            "pass": sample_gate,
            "retained_market_task_ids": retained_markets,
            "retained_candidate_ids": retained_families,
            "retained_control_ids": retained_controls,
        },
        "task_results": task_results,
        "decision": (
            "freeze is authorized"
            if ready
            else (
                "revise using annotations only and collect fresh ratings"
                if args.round == 1
                else "stop: a clean instrument could not be constructed after two rounds"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("coverage", "agreement", "retention", "decision")}, indent=2))
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
