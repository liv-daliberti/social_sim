#!/usr/bin/env python3
"""Render the current completed-panel Stage 3 item-level appendix audit."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exp1_prospective.stage3_materials_annotation import analyze_annotations as base


HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = HERE / "data/exports/interim_summary_20260827T204456Z.json"
DEFAULT_RESPONSES = HERE / "data/exports/registered_20260827T204456Z.csv"
DEFAULT_GENERATED = HERE / "generated_v6"
SHORT_LABELS = {
    "s3mat_8fc94e72b31731": "Hormuz traffic normal by July 31",
    "s3mat_5b895f10253115": "Iran reaches World Cup knockouts",
    "s3mat_46ce593e3c7968": "Israel strikes Yemen by June 30",
    "s3mat_752b90b3bb848e": "Netanyahu exits election by July 31",
    "s3mat_bfbaddb254c375": r"Anthropic valuation reaches \$1.1T",
    "s3mat_88672279d8a344": "Oladokun starts Chiefs Week 1",
    "s3mat_0e95090ec1ad85": r"S\&P 500 reaches 7,700 in June",
    "s3mat_2692ad43e3277e": "Trump unfreezes Iranian assets",
    "s3mat_576951368ad7e8": r"May durable-goods orders: 0--2\%",
    "s3mat_f8f9b0b657d63f": r"China Q2 growth: 4.6--4.9\%",
    "s3mat_f2c73811af53e8": "Eisenkot joins Bennett--Lapid alliance",
    "s3mat_7618d719cec3dc": "Russia enters Orikhiv by July 31",
    "s3mat_4d536ae0480586": "Romanian PM is independent/technocrat",
    "s3mat_fcf3cd9e4a753a": "Russia captures Kostyantynivka",
    "s3mat_b0dd3de045bf6d": "At least four June ChatGPT outages",
    "s3mat_1fee36f107823d": "El-Sayed wins Michigan primary",
    "s3mat_4abcd4e6c78969": "Little is MN-02 Democratic nominee",
    "s3mat_91a2619a1d51e7": r"OpenAI IPO close: \$1.25--1.5T",
}

DIRECTION_LABELS = {
    "more_likely": "More",
    "less_likely": "Less",
    "no_material_effect": "No effect",
    "ambiguous": "Ambig.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def median_text(values: list[int]) -> str:
    value = statistics.median(values)
    return f"{value:g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument(
        "--private-key", type=Path, default=DEFAULT_GENERATED / "private_key.jsonl"
    )
    parser.add_argument(
        "--public-items", type=Path, default=DEFAULT_GENERATED / "public_items.jsonl"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_GENERATED / "interim_item_validation_audit.json",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("paper/tables/exp1_stage3_item_validation.tex"),
    )
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    included = tuple(snapshot["included_completed_reviewers"])
    if len(included) < 2:
        raise SystemExit(f"too few included reviewers: {included}")
    reviewer_count = len(included)
    one_dissent_threshold = reviewer_count - 1

    responses = [
        row for row in read_csv(args.responses) if row["reviewer_id"] in included
    ]
    private_rows = base.read_jsonl(args.private_key)
    public_rows = base.read_jsonl(args.public_items)
    private = {str(row["item_id"]): row for row in private_rows}
    public = {str(row["item_id"]): row for row in public_rows}
    if set(private) != set(public) or set(private) != set(SHORT_LABELS):
        raise SystemExit("item sets differ across key, public packet, and display labels")

    expected_pairs = {(reviewer, item) for reviewer in included for item in private}
    observed_pairs = {(row["reviewer_id"], row["item_id"]) for row in responses}
    if observed_pairs != expected_pairs or len(responses) != len(expected_pairs):
        raise SystemExit("included reviewer-item coverage is incomplete or duplicated")

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        item_id = row["item_id"]
        if row["conditional_direction"] not in base.DIRECTION_VALUES:
            raise SystemExit(f"invalid direction for {item_id}")
        if row["usable_premise"] not in base.TERNARY_VALUES:
            raise SystemExit(f"invalid usability response for {item_id}")
        parsed = dict(row)
        for field in ("direction_confidence", "clarity", "plausibility"):
            parsed[field] = int(row[field])
            if parsed[field] not in range(1, 6):
                raise SystemExit(f"invalid {field} for {item_id}")
        by_item[item_id].append(parsed)

    audit_rows = []
    tex_rows = []
    for index, public_row in enumerate(public_rows, start=1):
        item_id = str(public_row["item_id"])
        rows = by_item[item_id]
        key = str(private[item_id]["expected_conditional_direction"])
        counts = Counter(row["conditional_direction"] for row in rows)
        maximum = max(counts.values())
        modes = sorted(direction for direction, count in counts.items() if count == maximum)
        majority = modes[0] if len(modes) == 1 else "ambiguous"
        correct = counts[key]
        usable = sum(row["usable_premise"] == "yes" for row in rows)
        confidence = statistics.median(row["direction_confidence"] for row in rows)
        clarity = statistics.median(row["clarity"] for row in rows)
        plausibility = statistics.median(row["plausibility"] for row in rows)
        failures = []
        if correct < one_dissent_threshold:
            failures.append("D")
        if confidence < 4:
            failures.append("C")
        if clarity < 4:
            failures.append("L")
        if plausibility < 3:
            failures.append("P")
        if usable < one_dissent_threshold:
            failures.append("U")
        passed = not failures
        gate = "Pass" if passed else f"Fail ({','.join(failures)})"
        audit_rows.append(
            {
                "item_number": index,
                "item_id": item_id,
                "question": public_row["question"],
                "registered_direction": key,
                "majority_direction": majority,
                "direction_correct_n": correct,
                "direction_denom": reviewer_count,
                "median_direction_confidence": confidence,
                "median_clarity": clarity,
                "median_plausibility": plausibility,
                "usable_premise_yes_n": usable,
                "usable_premise_denom": reviewer_count,
                "passes_interim_gate": passed,
                "failure_codes": failures,
            }
        )
        tex_rows.append(
            f"{index}. {SHORT_LABELS[item_id]} & {DIRECTION_LABELS[key]} & "
            f"{DIRECTION_LABELS[majority]} & {correct}/{reviewer_count} & "
            f"{median_text([row['direction_confidence'] for row in rows])}/"
            f"{median_text([row['clarity'] for row in rows])}/"
            f"{median_text([row['plausibility'] for row in rows])} & "
            f"{usable}/{reviewer_count} \\\\"
        )

    correct_total = sum(row["direction_correct_n"] for row in audit_rows)
    usable_total = sum(row["usable_premise_yes_n"] for row in audit_rows)
    passing_total = sum(row["passes_interim_gate"] for row in audit_rows)
    majority_exact = sum(
        row["majority_direction"] == row["registered_direction"] for row in audit_rows
    )
    pooled = snapshot["pooled_included"]
    panel = snapshot["panel_agreement"]
    if (
        pooled["direction_correct_n"] != correct_total
        or pooled["premise_usable_yes_n"] != usable_total
        or panel["items_passing_interim_one_dissent_gate"] != passing_total
        or panel["majority_exact_n"] != majority_exact
    ):
        raise SystemExit("generated item totals disagree with frozen snapshot")

    audit = {
        "source_snapshot": str(args.snapshot),
        "source_csv_sha256": snapshot["source_csv_sha256"],
        "included_reviewers": list(included),
        "item_count": len(audit_rows),
        "direction_correct_n": correct_total,
        "direction_denom": reviewer_count * len(audit_rows),
        "premise_usable_yes_n": usable_total,
        "premise_usable_denom": reviewer_count * len(audit_rows),
        "passing_item_count": passing_total,
        "majority_exact_n": majority_exact,
        "majority_denom": 18,
        "items": audit_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    latex = "\n".join(
        [
            r"\begin{tabular}{@{}p{.40\linewidth}llccc@{}}",
            r"\toprule",
            r"Question & Key & Majority & Agree & C/L/P & Usable \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(latex, encoding="utf-8")

    print(
        f"rendered {len(audit_rows)} items; "
        f"direction={correct_total}/{reviewer_count * len(audit_rows)}; "
        f"usable={usable_total}/{reviewer_count * len(audit_rows)}; "
        f"pass={passing_total}/18; "
        f"majority={majority_exact}/18"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
