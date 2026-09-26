#!/usr/bin/env python3
"""Exploratory direction companion and matched numeric-update diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from exp1_prospective.context_reversal import analyze as numeric
from exp1_prospective.context_reversal import direction_design as design
from exp1_prospective.context_reversal import run_direction_local as runner
from exp1_prospective.context_reversal import run_local as common


def expected_direction(context: str, condition: str) -> str | None:
    if condition != "new_news":
        return "unchanged"
    return {"positive": "increase", "negative": "decrease", "broken": "unchanged", "masked": None}[context]


def direction_value(row: dict | None) -> tuple[str | None, str]:
    if row is None:
        return None, "missing_record"
    if row["status"] != "ok":
        return None, row["status"]
    try:
        parsed = runner.parse_direction(row.get("raw"))
    except ValueError:
        return None, "invalid_direction"
    if row.get("direction") != parsed:
        return None, "invalid_direction"
    return parsed, "ok"


def index_responses(responses: list[dict], units: dict[str, dict], model_key: str) -> tuple[dict, int]:
    indexed, seen, ignored = {}, set(), 0
    for row in responses:
        if not all(isinstance(row.get(field), str) and row[field] for field in ("model_key", "trial_id", "condition")):
            raise ValueError("Direction response is missing its model/trial/condition key")
        key = (row["model_key"], row["trial_id"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate direction response: {key}")
        seen.add(key)
        if row["model_key"] != model_key:
            ignored += 1
            continue
        if row["trial_id"] not in units or row["condition"] not in design.CONDITIONS:
            raise ValueError(f"Direction response is outside the frozen design: {key}")
        unit = units[row["trial_id"]]
        if row.get("stage") != "direction" or row.get("status") not in runner.STATUSES:
            raise ValueError(f"Invalid direction response stage/status: {key}")
        for field in design.METADATA:
            if row.get(field) != unit[field] or (field == "repeat" and type(row[field]) is not int):
                raise ValueError(f"Direction response metadata mismatch: {key}/{field}")
        for field, expected in (("parent_plan_sha256", unit["parent_plan_sha256"]),
                                ("prompt", unit["direction_prompts"][row["condition"]]),
                                ("prompt_sha256", unit["direction_prompt_sha256"][row["condition"]])):
            if field in row and row[field] != expected:
                raise ValueError(f"Direction response provenance mismatch: {key}/{field}")
        indexed[row["trial_id"], row["condition"]] = row
    return indexed, ignored


def _cross_table(rows: list[dict]) -> dict:
    counts = {"both_correct": 0, "classification_correct_numeric_incorrect": 0,
              "classification_incorrect_numeric_correct": 0, "both_incorrect": 0}
    for row in rows:
        classification, number = row["classification_correct"], row["numeric_update_correct"]
        key = ("both_correct" if classification and number else "classification_correct_numeric_incorrect" if classification
               else "classification_incorrect_numeric_correct" if number else "both_incorrect")
        counts[key] += 1
    return {"n_planned": len(rows), **counts,
            "direction_invalid_or_missing": sum(not row["classification_valid"] for row in rows),
            "numeric_invalid_or_missing": sum(not row["numeric_update_valid"] for row in rows),
            "denominator_policy": "All planned scored rows; missing/invalid/unclear classifications and missing/invalid numeric updates fail."}


def probability_join(units: list[dict], indexed: dict, model_key: str,
                     probability_design: list[dict], probability_responses: list[dict],
                     probability_source_design: Path | None = None) -> dict:
    parent_units = numeric.validate_design(probability_design)
    if set(parent_units) != {u["trial_id"] for u in units}:
        raise ValueError("Probability design must be the exact single-repeat parent plan")
    for unit in units:
        parent = parent_units[unit["trial_id"]]
        if common.text_sha256(common.canonical_json(parent)) != unit["parent_unit_sha256"]:
            raise ValueError("Probability design differs from frozen parent unit")
        context, messages = design.extract_visible_text(parent)
        if context != unit["scenario_text"] or messages != unit["messages"]:
            raise ValueError("Probability and direction visible texts differ")
    # Existing local numeric runs may contain repeats 1 and 2; only exact parent repeat 0 is matched.
    selected = [r for r in probability_responses if r.get("model_key") == model_key and r.get("repeat") == 0]
    numeric_index, _ = numeric.index_responses(selected, parent_units, model_key)
    source_hash = units[0]["parent_plan_sha256"]
    if probability_source_design is not None:
        source_hash = common.file_sha256(probability_source_design)
        source_units = {row["trial_id"]: row for row in common.read_units(probability_source_design)}
        for trial, parent in parent_units.items():
            if source_units.get(trial) != parent:
                raise ValueError("Probability source design does not contain the exact frozen parent units")
    for row in selected:
        if row.get("input_sha256") != source_hash:
            raise ValueError("Probability response input hash differs from its source design")
        parent = parent_units[row["trial_id"]]
        baseline = numeric_index.get((row["trial_id"], "baseline"))
        initial, _ = numeric.probability(baseline)
        if row["condition"] == "baseline":
            prompt = parent["baseline_prompt"]
        else:
            prompt = common.render_update(parent, row["condition"], initial) if initial is not None else None
            if row.get("prior_probability") != initial:
                raise ValueError("Probability update prior differs from its existing baseline")
        if row.get("prompt") != prompt or row.get("prompt_sha256") != (common.text_sha256(prompt) if prompt is not None else None):
            raise ValueError("Probability response prompt differs from frozen parent scenario and own prior")
        if row["status"] == "ok" and common.parse_probability(row.get("raw", "")) != row.get("probability"):
            raise ValueError("Probability response raw completion and parsed value differ")
    rows = []
    for unit in units:
        trial, context = unit["trial_id"], unit["context_id"]
        initial, initial_reason = numeric.probability(numeric_index.get((trial, "baseline")))
        for condition in design.CONDITIONS:
            label, label_reason = direction_value(indexed.get((trial, condition)))
            value, value_reason = numeric.probability(numeric_index.get((trial, condition)))
            delta = None if initial is None or value is None else value - initial
            target = expected_direction(context, condition)
            actual_numeric_direction = None if delta is None else "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
            rows.append({
                **{key: unit[key] for key in design.METADATA}, "condition": condition,
                "expected_direction": target, "classification": label, "classification_status": label_reason,
                "classification_valid": label is not None,
                "classification_correct": None if target is None else label == target,
                "baseline_probability": initial, "baseline_status": initial_reason,
                "update_probability": value, "update_status": value_reason,
                "raw_update_pp": None if delta is None else 100 * delta,
                "numeric_direction": actual_numeric_direction, "numeric_update_valid": delta is not None,
                "numeric_update_correct": None if target is None else actual_numeric_direction == target,
                "baseline_at_zero": None if initial is None else initial == 0,
                "baseline_at_one": None if initial is None else initial == 1,
                "update_at_zero": None if value is None else value == 0,
                "update_at_one": None if value is None else value == 1,
                "expected_direction_blocked_by_baseline_endpoint": None if initial is None or target not in {"increase", "decrease"}
                    else (target == "increase" and initial == 1) or (target == "decrease" and initial == 0),
            })
    cross_tabs = {context: {condition: None if expected_direction(context, condition) is None else _cross_table([
        row for row in rows if row["context_id"] == context and row["condition"] == condition
    ]) for condition in design.CONDITIONS} for context in design.CONTEXTS}
    signed_rows = [r for r in rows if r["condition"] == "new_news" and r["context_id"] in {"positive", "negative"}]
    endpoint_rows = [r for r in rows if r["condition"] == "new_news"]
    flags = {flag: {"count": sum(r[flag] is True for r in endpoint_rows),
                    "observed": sum(r[flag] is not None for r in endpoint_rows), "planned": len(endpoint_rows)}
             for flag in ("baseline_at_zero", "baseline_at_one", "update_at_zero", "update_at_one",
                          "expected_direction_blocked_by_baseline_endpoint")}
    return {
        "status": "exploratory_matched_diagnostic", "matched_repeat": 0,
        "interpretation": "Matches the existing repeat-0 numeric responses; does not replace the original primary numeric analysis.",
        "numeric_correctness_rule": "Strict raw update sign for positive/negative new news; exact zero raw update for broken and controls; masked new news is unscored.",
        "excluded_probability_records": len(probability_responses) - len(selected),
        "cross_tabs": cross_tabs, "signed_new_news_pooled": _cross_table(signed_rows),
        "endpoint_flags_new_news_units": flags,
        "signed_new_news_by_endpoint_constraint": {
            "blocked": _cross_table([r for r in signed_rows if r["expected_direction_blocked_by_baseline_endpoint"] is True]),
            "unblocked": _cross_table([r for r in signed_rows if r["expected_direction_blocked_by_baseline_endpoint"] is False]),
            "baseline_missing": _cross_table([r for r in signed_rows if r["expected_direction_blocked_by_baseline_endpoint"] is None]),
        },
        "rows": rows,
    }


def analyze(direction_design: list[dict], responses: list[dict], model_key: str, *,
            probability_design: list[dict] | None = None, probability_responses: list[dict] | None = None,
            probability_source_design: Path | None = None, bootstrap_draws: int = numeric.DEFAULT_BOOTSTRAP_DRAWS, seed: int = numeric.DEFAULT_SEED) -> dict:
    if not model_key:
        raise ValueError("model_key is required")
    units = design.validate_units(direction_design)
    indexed, ignored = index_responses(responses, units, model_key)
    families = sorted({unit["family_id"] for unit in direction_design})
    bootstrap = numeric.FamilyBootstrap(families, bootstrap_draws, seed)
    cells, samples, pooled, paired = {}, {}, [], defaultdict(dict)
    total_coverage = Counter()
    confusion = defaultdict(Counter)
    for context in design.CONTEXTS:
        cells[context] = {}
        for condition in design.CONDITIONS:
            target, coverage, labels, accuracy, unclear = expected_direction(context, condition), Counter(), Counter(), [], []
            for unit in direction_design:
                if unit["context_id"] != context:
                    continue
                row = indexed.get((unit["trial_id"], condition))
                label, reason = direction_value(row)
                observed_label = label if label is not None else reason
                labels[observed_label] += 1
                coverage.update({"expected": 1, "planned": 1, "present": int(row is not None), "valid": int(label is not None),
                                 "missing_record": int(row is None), "invalid_present": int(row is not None and label is None),
                                 "unclear": int(label == "unclear")})
                coverage[f"reason:{reason}"] += 1
                if target is not None:
                    accuracy.append(numeric.Sample(unit["family_id"], None if label is None else float(label == target)))
                    confusion[target][observed_label] += 1
                unclear.append(numeric.Sample(unit["family_id"], None if label is None else float(label == "unclear")))
                if condition == "new_news":
                    paired[unit["family_id"]][context] = label
            cells[context][condition] = {"expected_direction": target, "coverage": dict(coverage),
                                        "label_counts": dict(labels), "accuracy": bootstrap.summarize(accuracy, binary=True) if target else None,
                                        "unclear_rate": bootstrap.summarize(unclear, binary=True)}
            samples[context, condition] = accuracy
            total_coverage.update(coverage)
        if context in {"positive", "negative"}:
            pooled.extend(samples[context, "new_news"])
    pair_samples = [numeric.Sample(family, None if signs["positive"] is None or signs["negative"] is None else
                                  float(signs["positive"] == "increase" and signs["negative"] == "decrease"))
                    for family, signs in paired.items()]
    join = None
    if (probability_design is None) != (probability_responses is None):
        raise ValueError("Supply both probability_design and probability_responses for the matched diagnostic")
    if probability_design is not None:
        join = probability_join(direction_design, indexed, model_key, probability_design, probability_responses, probability_source_design)
    return {
        "schema_version": "context_reversal_direction_analysis_v1", "analysis_status": "exploratory_direction_companion",
        "model_key": model_key, "n_families": len(families), "n_planned_units": len(direction_design),
        "n_planned_response_records": 3 * len(direction_design), "other_model_records_ignored": ignored,
        "material_status": sorted({u["material_status"] for u in direction_design}),
        "coverage": {"totals": dict(total_coverage)}, "by_context_condition": cells,
        "companion": {
            "direction_correct": {**{c: cells[c]["new_news"]["accuracy"] for c in ("positive", "negative")},
                                  "pooled": bootstrap.summarize(pooled, binary=True)},
            "paired_reversal": {"both_directions_correct": bootstrap.summarize(pair_samples, binary=True)},
            "broken_new_news_unchanged": cells["broken"]["new_news"]["accuracy"],
            "controls_unchanged": bootstrap.summarize([sample for (c, condition), values in samples.items()
                                                       if condition != "new_news" for sample in values], binary=True),
        },
        "confusion_counts": {target: dict(counts) for target, counts in confusion.items()},
        "bootstrap": {"unit": "family_id", "draws": bootstrap_draws, "seed": seed,
                      "preserves": ["contexts", "three_independent_message_conditions", "paired_reversal"]},
        "interpretation": [
            "Three independent classifications per parent scenario; no numeric forecast is displayed or elicited.",
            "All planned rows remain in accuracy denominators; unclear, wrong direction, parse failure, and missing outputs fail.",
            "Unchanged is a failure for positive/negative new news; it is the target for broken new news and all controls.",
            "Masked new news has no expected target and is unscored; its two control messages target unchanged.",
            "Unclear rates use planned denominators and report missingness separately; missing is not imputed as unclear.",
            "Whole-family bootstrap intervals are descriptive; this companion does not replace the parent numeric primary outcomes.",
            "Existing material status is retained; user reports human review already completed and no new review is requested.",
        ],
        "exploratory_probability_join": join,
    }


def markdown_summary(report: dict) -> str:
    lines = [f"# Direction companion: {report['model_key']}", "",
             "Exploratory companion to the existing numeric forecasting protocol.", "",
             f"Planned: {report['n_families']} families, {report['n_planned_response_records']} independent direction responses.", "",
             "| Context | Message | Target | Correct / planned | Unclear | Invalid or missing |",
             "|---|---|---|---:|---:|---:|"]
    for context, conditions in report["by_context_condition"].items():
        for condition, cell in conditions.items():
            accuracy, coverage = cell["accuracy"], cell["coverage"]
            text = "unscored" if accuracy is None else f"{accuracy['n_success']} / {accuracy['n_planned']}"
            lines.append(f"| {context} | {condition} | {cell['expected_direction'] or 'none'} | {text} | {coverage['unclear']} | {coverage['missing_record'] + coverage['invalid_present']} |")
    paired = report["companion"]["paired_reversal"]["both_directions_correct"]
    lines.extend(["", f"Both directions correct: {paired['n_success']} / {paired['n_planned']} families; 95% family-bootstrap interval: {paired['ci95']}.", "",
                  "All planned trials remain in scored denominators. Unclear, unchanged for signed targets, parse failures, and missing results fail. Masked new news has no target."])
    join = report["exploratory_probability_join"]
    if join is not None:
        cross = join["signed_new_news_pooled"]
        lines.extend(["", "## Exploratory comparison with existing numeric updates", "",
                      "| | Numeric correct | Numeric incorrect or missing |", "|---|---:|---:|",
                      f"| Classification correct | {cross['both_correct']} | {cross['classification_correct_numeric_incorrect']} |",
                      f"| Classification incorrect, unclear, or missing | {cross['classification_incorrect_numeric_correct']} | {cross['both_incorrect']} |", "",
                      "Comparison uses existing repeat-0 responses and strict positive/negative raw update signs. Numeric zero updates fail signed scoring. Endpoint flags and per-cell confusion counts are retained in summary.json."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--probability-design", type=Path)
    parser.add_argument("--probability-responses", type=Path, nargs="+")
    parser.add_argument("--probability-source-design", type=Path, help="Original full design used to generate numeric responses; allows selecting its repeat 0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=numeric.DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=numeric.DEFAULT_SEED)
    args = parser.parse_args()
    if (args.probability_design is None) != (args.probability_responses is None):
        parser.error("Matched analysis requires --probability-design and --probability-responses")
    units = design.read_units(args.design)
    responses = [row for path in args.responses for row in numeric.read_jsonl(path)]
    plan_hash = common.file_sha256(args.design)
    for row in responses:
        if row.get("model_key") == args.model_key and row.get("input_sha256") != plan_hash:
            parser.error("Direction response input hash does not match the supplied design")
    parent = numeric.read_jsonl(args.probability_design) if args.probability_design else None
    numbers = [row for path in (args.probability_responses or []) for row in numeric.read_jsonl(path)]
    if args.probability_design and common.file_sha256(args.probability_design) != units[0]["parent_plan_sha256"]:
        parser.error("Probability design hash differs from frozen parent")
    report = analyze(units, responses, args.model_key, probability_design=parent,
                     probability_responses=numbers if parent is not None else None,
                     probability_source_design=args.probability_source_design, bootstrap_draws=args.bootstrap_draws, seed=args.seed)
    report["provenance"] = {"design": {"path": str(args.design), "sha256": plan_hash},
                            "responses": [{"path": str(p), "sha256": common.file_sha256(p)} for p in args.responses],
                            "probability_design": None if args.probability_design is None else
                                {"path": str(args.probability_design), "sha256": common.file_sha256(args.probability_design)},
                            "probability_source_design": None if args.probability_source_design is None else
                                {"path": str(args.probability_source_design), "sha256": common.file_sha256(args.probability_source_design)},
                            "probability_responses": [{"path": str(p), "sha256": common.file_sha256(p)} for p in (args.probability_responses or [])]}
    common.atomic_write(args.output / "summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    common.atomic_write(args.output / "summary.md", markdown_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
