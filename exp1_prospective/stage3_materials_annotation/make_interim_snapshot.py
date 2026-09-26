#!/usr/bin/env python3
"""Create an auditable snapshot of completed Stage 3 review packets.

The script applies the documented post-hoc straightlining rule uniformly to
every complete packet, retains excluded packets for the all-completed
sensitivity analysis, and reports the registered item-retention diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


CATEGORIES = ["more_likely", "less_likely", "no_material_effect", "ambiguous"]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cohen(a: list[str], b: list[str]) -> tuple[float, float]:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in CATEGORIES)
    return (
        (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    ), observed


def fleiss(counts: list[collections.Counter], categories: list[str]) -> float:
    n = sum(counts[0].values())
    item_count = len(counts)
    observed = (
        sum(
            (sum(value * value for value in item.values()) - n) / (n * (n - 1))
            for item in counts
        )
        / item_count
    )
    totals = collections.Counter()
    for item in counts:
        totals.update(item)
    expected = sum((totals[c] / (item_count * n)) ** 2 for c in categories)
    return (observed - expected) / (1 - expected)


def binomial_upper_tail(correct: int, total: int, chance: float = 1 / 3) -> float:
    return sum(
        math.comb(total, value) * chance**value * (1 - chance) ** (total - value)
        for value in range(correct, total + 1)
    )


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-version", default="stage3_materials_annotation_v8")
    parser.add_argument(
        "--final-descriptive",
        action="store_true",
        help=(
            "Close the descriptive materials review at the completed roster, "
            "remove obsolete recruitment-gate fields, and require at least six "
            "quality-eligible reviewers."
        ),
    )
    args = parser.parse_args()

    status = json.loads(args.status.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    key_rows = read_jsonl(args.private_key)
    key = {str(row["item_id"]): row for row in key_rows}
    items = sorted(key)
    with args.responses.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    by_reviewer: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_reviewer[row["reviewer_id"]].append(row)
    completed = sorted(
        reviewer
        for reviewer, reviewer_rows in by_reviewer.items()
        if len(reviewer_rows) == len(items)
    )
    if completed != sorted(status["completed_reviewers"]):
        raise SystemExit("CSV completion roster disagrees with status export")

    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for reviewer in completed:
        mapping = {row["item_id"]: row for row in by_reviewer[reviewer]}
        if set(mapping) != set(items) or len(mapping) != len(by_reviewer[reviewer]):
            raise SystemExit(f"incomplete or duplicated item coverage for {reviewer}")
        indexed[reviewer] = mapping

    cutoff = int(policy["rule"]["exclude_if_any_single_response_selected_at_least"])
    excluded = []
    for reviewer in completed:
        pattern = collections.Counter(
            indexed[reviewer][item]["conditional_direction"] for item in items
        )
        if max(pattern.values()) >= cutoff:
            excluded.append(reviewer)
    included = [reviewer for reviewer in completed if reviewer not in excluded]
    if not included:
        raise SystemExit("quality rule excluded every completed reviewer")

    gold = {item: str(key[item]["expected_conditional_direction"]) for item in items}
    packet_class = {item: str(key[item]["registered_direction"]) for item in items}
    reviewers = {}
    for reviewer in completed:
        reviewer_rows = [indexed[reviewer][item] for item in items]
        labels = [row["conditional_direction"] for row in reviewer_rows]
        correct = sum(label == gold[item] for label, item in zip(labels, items))
        kappa, _ = cohen(labels, [gold[item] for item in items])
        starts = [
            parse_time(row["started_at"]) for row in reviewer_rows if row["started_at"]
        ]
        ends = [
            parse_time(row["submitted_at"])
            for row in reviewer_rows
            if row["submitted_at"]
        ]
        reviewers[reviewer] = {
            "cohen_kappa_vs_registered_key": kappa,
            "completion_window_seconds": (max(ends) - min(starts)).total_seconds(),
            "direction_correct_n": correct,
            "direction_correct_rate": correct / len(items),
            "direction_denom": len(items),
            "direction_response_pattern": dict(
                sorted(collections.Counter(labels).items())
            ),
            "median_clarity": statistics.median(
                int(row["clarity"]) for row in reviewer_rows
            ),
            "median_direction_confidence": statistics.median(
                int(row["direction_confidence"]) for row in reviewer_rows
            ),
            "median_plausibility": statistics.median(
                int(row["plausibility"]) for row in reviewer_rows
            ),
            "one_sided_exact_binomial_p_vs_one_third": binomial_upper_tail(
                correct, len(items)
            ),
            "premise_usable_yes_n": sum(
                row["usable_premise"] == "yes" for row in reviewer_rows
            ),
            "quality_excluded": reviewer in excluded,
        }

    def majority(item: str) -> str:
        counts = collections.Counter(
            indexed[r][item]["conditional_direction"] for r in included
        ).most_common()
        return (
            "ambiguous"
            if len(counts) > 1 and counts[0][1] == counts[1][1]
            else counts[0][0]
        )

    majorities = {item: majority(item) for item in items}
    counts = [
        collections.Counter(indexed[r][item]["conditional_direction"] for r in included)
        for item in items
    ]
    folded = [
        collections.Counter(
            "no_material_effect"
            if indexed[r][item]["conditional_direction"] == "ambiguous"
            else indexed[r][item]["conditional_direction"]
            for r in included
        )
        for item in items
    ]
    pairs = [
        cohen(
            [indexed[a][item]["conditional_direction"] for item in items],
            [indexed[b][item]["conditional_direction"] for item in items],
        )[0]
        for a, b in itertools.combinations(included, 2)
    ]
    four_kappa = fleiss(counts, CATEGORIES)
    majority_kappa, _ = cohen(
        [majorities[item] for item in items], [gold[item] for item in items]
    )

    item_gates = []
    for item in items:
        item_rows = [indexed[r][item] for r in included]
        direction_match = sum(
            row["conditional_direction"] == gold[item] for row in item_rows
        )
        usable = sum(row["usable_premise"] == "yes" for row in item_rows)
        record = {
            "item_id": item,
            "direction_match": direction_match,
            "median_confidence": statistics.median(
                int(row["direction_confidence"]) for row in item_rows
            ),
            "median_clarity": statistics.median(
                int(row["clarity"]) for row in item_rows
            ),
            "median_plausibility": statistics.median(
                int(row["plausibility"]) for row in item_rows
            ),
            "premise_usable": usable,
        }
        record["passes"] = (
            direction_match >= len(included) - 1
            and record["median_confidence"] >= 4
            and record["median_clarity"] >= 4
            and record["median_plausibility"] >= 3
            and usable >= len(included) - 1
        )
        item_gates.append(record)

    included_correct = sum(reviewers[r]["direction_correct_n"] for r in included)
    included_usable = sum(reviewers[r]["premise_usable_yes_n"] for r in included)
    denominator = len(included) * len(items)
    by_class = {}
    for category in ("pro_H1", "anti_H1", "orthogonal"):
        subset = [item for item in items if packet_class[item] == category]
        by_class[category] = {
            "reviewer_correct": sum(
                indexed[r][item]["conditional_direction"] == gold[item]
                for r in included
                for item in subset
            ),
            "reviewer_denom": len(included) * len(subset),
            "majority_correct": sum(majorities[item] == gold[item] for item in subset),
            "majority_denom": len(subset),
        }

    snapshot = {
        "analysis": "interim_quality_filtered_completed_reviewers",
        "protocol_version": args.protocol_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "completed_reviewers": completed,
        "excluded_reviewers": excluded,
        "included_completed_reviewers": included,
        "partial_reviewers": {
            reviewer: count
            for reviewer, count in sorted(status["ratings_by_reviewer"].items())
            if 0 < count < len(items)
        },
        "final_gate_available": len(completed) == int(status["expected_reviewers"]),
        "reason_final_gate_unavailable": (
            None
            if len(completed) == int(status["expected_reviewers"])
            else "The nine-reviewer roster is incomplete; this snapshot summarizes completed packets only."
        ),
        "status": status,
        "reviewers": reviewers,
        "pooled_included": {
            "direction_correct_n": included_correct,
            "direction_denom": denominator,
            "direction_correct_rate": included_correct / denominator,
            "premise_usable_yes_n": included_usable,
            "premise_usable_denom": denominator,
            "premise_usable_rate": included_usable / denominator,
        },
        "all_completed_sensitivity": {
            "direction_correct_n": sum(
                reviewers[r]["direction_correct_n"] for r in completed
            ),
            "direction_denom": len(completed) * len(items),
            "direction_correct_rate": sum(
                reviewers[r]["direction_correct_n"] for r in completed
            )
            / (len(completed) * len(items)),
        },
        "panel_agreement": {
            "fleiss_kappa_four_category": four_kappa,
            "fleiss_kappa_ambiguous_folded_into_null": fleiss(folded, CATEGORIES[:3]),
            "mean_pairwise_cohen_kappa": sum(pairs) / len(pairs),
            "pairwise_cohen_kappa_range": [min(pairs), max(pairs)],
            "majority_exact_n": sum(majorities[item] == gold[item] for item in items),
            "majority_denom": len(items),
            "majority_kappa_vs_registered_key": majority_kappa,
            "direction_only_items_with_at_most_one_dissent": sum(
                row["direction_match"] >= len(included) - 1 for row in item_gates
            ),
            "unanimous_direction_items": sum(
                row["direction_match"] == len(included) for row in item_gates
            ),
            "items_passing_interim_one_dissent_gate": sum(
                row["passes"] for row in item_gates
            ),
            "by_packet_class": by_class,
        },
    }
    if args.final_descriptive:
        partial = snapshot["partial_reviewers"]
        target_met = len(included) >= 6 and not partial
        if not target_met:
            raise SystemExit(
                "final descriptive summary requires at least six included complete "
                "reviewers and no partially completed packets"
            )
        snapshot["analysis"] = "final_descriptive_materials_review"
        snapshot["analysis_status"] = "complete"
        snapshot["collection"] = {
            "closed": True,
            "completed_reviewer_count": len(completed),
            "included_reviewer_count": len(included),
            "excluded_reviewer_count": len(excluded),
            "minimum_included_reviewers": 6,
            "descriptive_target_met": True,
        }
        snapshot["source_status_sha256"] = hashlib.sha256(
            args.status.read_bytes()
        ).hexdigest()
        snapshot.pop("final_gate_available")
        snapshot.pop("reason_final_gate_unavailable")
        snapshot.pop("status")
        snapshot["panel_agreement"].pop("items_passing_interim_one_dissent_gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output}: {len(completed)} complete, {len(included)} included, "
        f"{included_correct}/{denominator} direction-correct"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
