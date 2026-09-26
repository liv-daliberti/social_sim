#!/usr/bin/env python3
"""Validate and summarize the Stage 3 materials annotations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_GENERATED = HERE / "generated_v6"
DIRECTION_VALUES = (
    "more_likely",
    "less_likely",
    "no_material_effect",
    "ambiguous",
)
TERNARY_VALUES = ("yes", "unclear", "no")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fleiss_kappa(rows_by_item: dict[str, list[dict[str, Any]]]) -> float:
    if not rows_by_item:
        return float("nan")
    n_values = {len(rows) for rows in rows_by_item.values()}
    if len(n_values) != 1:
        raise ValueError("Fleiss' kappa requires equal ratings per item")
    n = next(iter(n_values))
    if n < 2:
        raise ValueError("Fleiss' kappa requires at least two ratings per item")
    category_totals = Counter()
    agreement = []
    for rows in rows_by_item.values():
        counts = Counter(row["conditional_direction"] for row in rows)
        category_totals.update(counts)
        agreement.append(
            (sum(value * value for value in counts.values()) - n) / (n * (n - 1))
        )
    item_count = len(rows_by_item)
    p_bar = sum(agreement) / item_count
    p_expected = sum(
        (category_totals[category] / (item_count * n)) ** 2
        for category in DIRECTION_VALUES
    )
    if p_expected == 1:
        return 1.0 if p_bar == 1 else float("nan")
    return (p_bar - p_expected) / (1 - p_expected)


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def validate_and_join(
    responses: list[dict[str, str]],
    assignments: dict[str, Any],
    private_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    private = {str(row["item_id"]): row for row in private_rows}
    expected = {
        (str(assignment["reviewer_id"]), str(item_id))
        for assignment in assignments["assignments"]
        for item_id in assignment["item_ids_in_order"]
    }
    observed = [(row.get("reviewer_id", ""), row.get("item_id", "")) for row in responses]
    if len(observed) != len(set(observed)):
        raise SystemExit("duplicate reviewer-item response")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise SystemExit(f"response coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    expected_count = int(assignments["expected_rating_count"])
    if len(responses) != expected_count:
        raise SystemExit(
            f"expected {expected_count} responses, found {len(responses)}"
        )

    joined = []
    for row in responses:
        item_id = row["item_id"]
        if item_id not in private:
            raise SystemExit(f"unknown item_id: {item_id}")
        if row.get("consent_confirmed") != "yes":
            raise SystemExit(f"missing consent confirmation: {row['reviewer_id']} {item_id}")
        if row.get("conditional_direction") not in DIRECTION_VALUES:
            raise SystemExit(f"invalid conditional_direction: {row}")
        if row.get("usable_premise") not in TERNARY_VALUES:
            raise SystemExit(f"invalid usable_premise: {row}")
        numeric: dict[str, int] = {}
        for field in ("direction_confidence", "clarity", "plausibility"):
            try:
                value = int(row.get(field, ""))
            except ValueError as exc:
                raise SystemExit(f"{field} is not an integer: {row}") from exc
            if value not in range(1, 6):
                raise SystemExit(f"{field} is outside 1--5: {row}")
            numeric[field] = value
        key = private[item_id]
        joined.append(
            {
                **row,
                **numeric,
                "registered_direction": key["registered_direction"],
                "expected_conditional_direction": key["expected_conditional_direction"],
                "direction_correct": (
                    row["conditional_direction"] == key["expected_conditional_direction"]
                ),
            }
        )
    return joined


def summarize(joined: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_item[str(row["item_id"])].append(row)
        by_direction[str(row["registered_direction"])].append(row)

    item_results = []
    for item_id, rows in sorted(by_item.items()):
        reviewer_count = len({row["reviewer_id"] for row in joined})
        if len(rows) != reviewer_count:
            raise SystemExit(
                f"{item_id}: expected {reviewer_count} ratings, found {len(rows)}"
            )
        agreement_required = reviewer_count - 1
        correct = sum(bool(row["direction_correct"]) for row in rows)
        usable = sum(row["usable_premise"] == "yes" for row in rows)
        confidence = statistics.median(row["direction_confidence"] for row in rows)
        clarity = statistics.median(row["clarity"] for row in rows)
        plausibility = statistics.median(row["plausibility"] for row in rows)
        passed = (
            correct >= agreement_required
            and confidence >= 4
            and clarity >= 4
            and plausibility >= 3
            and usable >= agreement_required
        )
        item_results.append(
            {
                "item_id": item_id,
                "registered_direction": rows[0]["registered_direction"],
                "direction_correct_n": correct,
                "direction_correct_denom": reviewer_count,
                "median_direction_confidence": confidence,
                "median_clarity": clarity,
                "median_plausibility": plausibility,
                "usable_premise_yes_n": usable,
                "passes_item_gate": passed,
            }
        )

    kappa = fleiss_kappa(by_item)
    passing_items = sum(row["passes_item_gate"] for row in item_results)
    overall_correct = sum(bool(row["direction_correct"]) for row in joined)
    direction_results = {}
    for direction, rows in sorted(by_direction.items()):
        correct = sum(bool(row["direction_correct"]) for row in rows)
        direction_results[direction] = {
            "ratings": len(rows),
            "direction_correct_n": correct,
            "direction_correct_rate": correct / len(rows),
            "usable_premise": distribution(rows, "usable_premise"),
        }

    return {
        "version": "stage3_materials_annotation_v6",
        "rating_count": len(joined),
        "reviewer_count": len({row["reviewer_id"] for row in joined}),
        "item_count": len(by_item),
        "direction_correct_n": overall_correct,
        "direction_correct_rate": overall_correct / len(joined),
        "fleiss_kappa_conditional_direction": kappa,
        "passing_item_count": passing_items,
        "global_gate_pass": passing_items >= 15 and kappa >= 0.70,
        "numeric_distributions": {
            field: distribution(joined, field)
            for field in ("direction_confidence", "clarity", "plausibility")
        },
        "usable_premise": distribution(joined, "usable_premise"),
        "by_registered_direction": direction_results,
        "items": item_results,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 materials-annotation report",
        "",
        f"- Ratings: {summary['rating_count']}",
        f"- Annotators: {summary['reviewer_count']}",
        f"- Items: {summary['item_count']}",
        (
            "- Registered-direction agreement: "
            f"{summary['direction_correct_n']}/{summary['rating_count']} "
            f"({100 * summary['direction_correct_rate']:.1f}%)"
        ),
        (
            "- Fleiss' kappa: "
            f"{summary['fleiss_kappa_conditional_direction']:.3f}"
        ),
        f"- Items passing all gates: {summary['passing_item_count']}/18",
        f"- Global validation gate: {'PASS' if summary['global_gate_pass'] else 'FAIL'}",
        "",
        "## Item results",
        "",
        "| Item | Direction | Agreement | Confidence | Clarity | Plausibility | Premise usable | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["items"]:
        lines.append(
            "| {item_id} | {registered_direction} | "
            "{direction_correct_n}/{direction_correct_denom} | "
            "{median_direction_confidence:g} | {median_clarity:g} | "
            "{median_plausibility:g} | "
            "{usable_premise_yes_n}/{direction_correct_denom} | {gate} |".format(
                **item, gate="PASS" if item["passes_item_gate"] else "FAIL"
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--assignments", type=Path, default=DEFAULT_GENERATED / "assignments.json"
    )
    parser.add_argument(
        "--private-key", type=Path, default=DEFAULT_GENERATED / "private_key.jsonl"
    )
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_GENERATED / "annotation_report.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=DEFAULT_GENERATED / "annotation_report.md"
    )
    args = parser.parse_args()

    responses = read_csv(args.responses)
    assignments = json.loads(args.assignments.read_text(encoding="utf-8"))
    private_rows = read_jsonl(args.private_key)
    joined = validate_and_join(responses, assignments, private_rows)
    summary = summarize(joined)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        f"{'PASS' if summary['global_gate_pass'] else 'FAIL'}: "
        f"{summary['passing_item_count']}/18 items; "
        f"kappa={summary['fleiss_kappa_conditional_direction']:.3f}"
    )
    return 0 if summary["global_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
