#!/usr/bin/env python3
"""Render the post-hoc quality-filtered Stage 3 annotation analysis.

The frozen seven-reviewer coverage check is always run first. Raw responses are
never deleted. The quality rule is then applied uniformly, and both the filtered
analysis and the registered all-rater sensitivity are written.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exp1_prospective.stage3_materials_annotation import analyze_annotations as base


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated_v6"
DEFAULT_POLICY = GENERATED / "posthoc_exclusions.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def detect_exclusions(
    responses: list[dict[str, str]], policy: dict[str, Any]
) -> tuple[set[str], dict[str, dict[str, int]]]:
    rule = policy["rule"]
    required = int(rule["minimum_completed_items"])
    maximum = int(rule["maximum_items"])
    threshold = int(rule["exclude_if_any_single_response_selected_at_least"])
    if required != maximum:
        raise SystemExit("quality rule must operate only on complete reviewer packets")

    by_reviewer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in responses:
        by_reviewer[row["reviewer_id"]].append(row)

    excluded: set[str] = set()
    patterns: dict[str, dict[str, int]] = {}
    for reviewer_id, rows in sorted(by_reviewer.items()):
        if len(rows) != required:
            continue
        counts = Counter(row["conditional_direction"] for row in rows)
        patterns[reviewer_id] = dict(sorted(counts.items()))
        if max(counts.values(), default=0) >= threshold:
            excluded.add(reviewer_id)
    return excluded, patterns


def filtered_inputs(
    responses: list[dict[str, str]],
    assignments: dict[str, Any],
    excluded: set[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    filtered_responses = [
        row for row in responses if row["reviewer_id"] not in excluded
    ]
    filtered_assignments = dict(assignments)
    filtered_assignments["assignments"] = [
        row
        for row in assignments["assignments"]
        if row["reviewer_id"] not in excluded
    ]
    reviewer_count = len(filtered_assignments["assignments"])
    items_per_reviewer = int(assignments["items_per_reviewer"])
    filtered_assignments["reviewer_count"] = reviewer_count
    filtered_assignments["ratings_per_item"] = reviewer_count
    filtered_assignments["expected_rating_count"] = (
        reviewer_count * items_per_reviewer
    )
    return filtered_responses, filtered_assignments


def attach_audit_metadata(
    summary: dict[str, Any],
    *,
    analysis_set: str,
    raw_rating_count: int,
    excluded: set[str],
    policy: dict[str, Any],
    patterns: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return {
        **summary,
        "analysis_set": analysis_set,
        "raw_rating_count": raw_rating_count,
        "excluded_reviewers": sorted(excluded),
        "exclusion_policy_version": policy["version"],
        "exclusion_decision_timing": policy["decision_timing"],
        "completed_reviewer_response_patterns": patterns,
    }


def audited_markdown(summary: dict[str, Any]) -> str:
    header = [
        "# Analysis-set audit",
        "",
        f"- Analysis set: {summary['analysis_set']}",
        f"- Raw ratings retained: {summary['raw_rating_count']}",
        "- Excluded reviewers: "
        + (", ".join(summary["excluded_reviewers"]) or "none"),
        f"- Exclusion timing: {summary['exclusion_decision_timing']}",
        "",
    ]
    return "\n".join(header) + base.render_markdown(summary)


def write_report(
    summary: dict[str, Any], output_json: Path, output_md: Path
) -> None:
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(audited_markdown(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--assignments", type=Path, default=GENERATED / "assignments.json"
    )
    parser.add_argument(
        "--private-key", type=Path, default=GENERATED / "private_key.jsonl"
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--filtered-json",
        type=Path,
        default=GENERATED / "annotation_report_quality_filtered.json",
    )
    parser.add_argument(
        "--filtered-md",
        type=Path,
        default=GENERATED / "annotation_report_quality_filtered.md",
    )
    parser.add_argument(
        "--sensitivity-json",
        type=Path,
        default=GENERATED / "annotation_report_all_raters_sensitivity.json",
    )
    parser.add_argument(
        "--sensitivity-md",
        type=Path,
        default=GENERATED / "annotation_report_all_raters_sensitivity.md",
    )
    args = parser.parse_args()

    responses = read_csv(args.responses)
    assignments = json.loads(args.assignments.read_text(encoding="utf-8"))
    private_rows = base.read_jsonl(args.private_key)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))

    # Fail closed on the complete frozen seven-reviewer roster before filtering.
    all_joined = base.validate_and_join(responses, assignments, private_rows)
    excluded, patterns = detect_exclusions(responses, policy)
    initial_trigger = str(policy["initial_trigger"]["reviewer_id"])
    if initial_trigger not in excluded:
        raise SystemExit(
            f"recorded trigger {initial_trigger} does not satisfy the coded rule"
        )

    quality_responses, quality_assignments = filtered_inputs(
        responses, assignments, excluded
    )
    quality_joined = base.validate_and_join(
        quality_responses, quality_assignments, private_rows
    )

    quality_summary = attach_audit_metadata(
        base.summarize(quality_joined),
        analysis_set="quality_filtered_post_hoc",
        raw_rating_count=len(responses),
        excluded=excluded,
        policy=policy,
        patterns=patterns,
    )
    sensitivity_summary = attach_audit_metadata(
        base.summarize(all_joined),
        analysis_set="registered_all_raters_sensitivity",
        raw_rating_count=len(responses),
        excluded=set(),
        policy=policy,
        patterns=patterns,
    )
    write_report(quality_summary, args.filtered_json, args.filtered_md)
    write_report(sensitivity_summary, args.sensitivity_json, args.sensitivity_md)

    print(
        f"quality-filtered={'PASS' if quality_summary['global_gate_pass'] else 'FAIL'} "
        f"({quality_summary['passing_item_count']}/18, "
        f"kappa={quality_summary['fleiss_kappa_conditional_direction']:.3f}); "
        f"excluded={','.join(sorted(excluded))}"
    )
    return 0 if quality_summary["global_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
