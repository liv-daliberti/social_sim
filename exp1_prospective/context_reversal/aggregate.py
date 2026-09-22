#!/usr/bin/env python3
"""Combine every prescribed model's descriptive development-pilot summary.

The fixed roster is never filtered for availability or performance. This tool
reads analysis summaries, not responses, and does not launch any model jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_KEYS = (
    "qwen2_5_72b_instruct",
    "llama3_1_70b_instruct",
    "qwen3_32b",
)
CONTEXTS = ("positive", "negative", "broken", "masked")
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
CAVEAT = (
    "Authored, unvalidated development materials. These are exploratory descriptive "
    "diagnostics, not confirmatory evidence or population estimates. All three "
    "prescribed models are included in fixed order; results must not be used to "
    "select a model."
)


def aggregate(results_root: Path) -> dict[str, Any]:
    paths = {model: results_root / model / "summary.json" for model in MODEL_KEYS}
    absent = [model for model, path in paths.items() if not path.is_file()]
    if absent:
        raise ValueError("Required model summaries are absent: " + ", ".join(absent))
    models: dict[str, Any] = {}
    design_hashes = set()
    plan_sizes = set()
    sources = {}
    for model, path in paths.items():
        raw = path.read_bytes()
        summary = json.loads(raw)
        if summary.get("model_key") != model:
            raise ValueError(f"Model key mismatch in {path}")
        if summary.get("schema_version") != "context_reversal_analysis_v1":
            raise ValueError(f"Unsupported analysis schema in {path}")
        if summary.get("material_status") != "authored_development_unvalidated" or summary.get("analysis_status") != "development_unvalidated_descriptive_only":
            raise ValueError(f"Summary is not an unvalidated development analysis: {path}")
        design_hash = summary.get("provenance", {}).get("design", {}).get("sha256")
        if not isinstance(design_hash, str) or not design_hash:
            raise ValueError(f"Missing design provenance in {path}")
        design_hashes.add(design_hash)
        plan_sizes.add((summary["n_families"], summary["n_planned_units"], summary["n_planned_response_records"]))
        models[model] = {
            "coverage": summary["coverage"],
            "baseline_probability_percent": summary["baseline_probability_percent"],
            "primary": summary["primary"],
            "drift_adjusted": summary["drift_adjusted"],
            "broken_raw_new_news": summary["raw_updates"]["broken"]["new_news"],
            "broken_stability": summary["broken_stability"],
            "control_updates": {
                context: {condition: summary["raw_updates"][context][condition] for condition in ("no_news", "repeated_news")}
                for context in CONTEXTS
            },
            "repeated_minus_no_news": summary["repeated_minus_no_news"],
            "bootstrap": summary["bootstrap"],
        }
        sources[model] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if len(design_hashes) != 1 or len(plan_sizes) != 1:
        raise ValueError("Prescribed model summaries do not share the same planned design")
    families, units, records = next(iter(plan_sizes))
    return {
        "schema_version": "context_reversal_development_aggregate_v1",
        "material_status": "authored_development_unvalidated",
        "analysis_status": "development_unvalidated_descriptive_only",
        "caveat": CAVEAT,
        "prescribed_models": list(MODEL_KEYS),
        "design_sha256": next(iter(design_hashes)),
        "n_families": families,
        "n_planned_units_per_model": units,
        "n_planned_response_records_per_model": records,
        "models": models,
        "source_summaries": sources,
    }


def estimate_text(metric: dict[str, Any], percent: bool = False) -> str:
    value = metric["estimate"]
    if value is None:
        return "unavailable"
    multiplier = 100 if percent else 1
    result = f"{value * multiplier:.2f}" + ("%" if percent else "")
    interval = metric["ci95"]
    if interval is not None:
        result += f" [{interval[0] * multiplier:.2f}, {interval[1] * multiplier:.2f}]"
    else:
        result += " [interval unavailable]"
    return result


def outcome_text(metric: dict[str, Any]) -> str:
    return f"{estimate_text(metric, percent=True)}; {metric['n_success']}/{metric['n_planned']}"


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Paired context-reversal development summary", "", report["caveat"], "",
        f"Each model has {report['n_families']} planned families, {report['n_planned_units_per_model']} context/repeat units, and {report['n_planned_response_records_per_model']} response records. The design hash is `{report['design_sha256']}`.", "",
        "## Coverage and paired outcomes", "",
        "Direction accuracy and paired reversal use all planned positive/negative trials or pairs. Zero updates and invalid or missing trials fail. Brackets show descriptive 95% family-bootstrap intervals; success/planned counts follow.", "",
        "| Model | Valid / planned responses | Missing / invalid records | Raw direction | Raw paired reversal | Drift-adjusted direction | Drift-adjusted paired reversal |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for model in MODEL_KEYS:
        source = report["models"][model]
        coverage = source["coverage"]["totals"]
        raw, adjusted = source["primary"], source["drift_adjusted"]
        lines.append(f"| {model} | {coverage['valid']} / {coverage['expected']} | {coverage['missing_record']} / {coverage['invalid_present']} | {outcome_text(raw['direction_correct']['pooled'])} | {outcome_text(raw['paired_reversal']['both_directions_correct'])} | {outcome_text(adjusted['direction_correct']['pooled'])} | {outcome_text(adjusted['paired_reversal']['both_directions_correct'])} |")
    lines.extend([
        "", "## Broken-link movement and stability", "",
        "Movement is in percentage points. Stability requires complete planned measurements, the signed-mean 90% interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance is not equivalence; opposing large revisions cannot establish stability.", "",
        "| Model | Raw mean absolute [95% interval] | Observed / planned | Raw criterion | Drift-adjusted mean absolute [95% interval] | Observed / planned | Adjusted criterion |",
        "|---|---|---:|---|---|---:|---|",
    ])
    for model in MODEL_KEYS:
        source = report["models"][model]
        raw = source["broken_raw_new_news"]["mean_absolute_pp"]
        adjusted = source["drift_adjusted"]["by_context"]["broken"]["mean_absolute_pp"]
        diagnostics = source["broken_stability"]
        raw_status, adjusted_status = diagnostics["raw_new_news"], diagnostics["drift_adjusted"]
        lines.append(f"| {model} | {estimate_text(raw)} | {raw['n_observed']} / {raw['n_planned']} | {raw_status['status']} ({raw_status['reason']}) | {estimate_text(adjusted)} | {adjusted['n_observed']} / {adjusted['n_planned']} | {adjusted_status['status']} ({adjusted_status['reason']}) |")
    lines.extend([
        "", "## No-news and repeated-news drift", "",
        "All movements are relative to the shared context-specific baseline, in percentage points. These magnitude summaries use observed complete measurements; missing forecasts are never zero-imputed. Masked contexts have no prespecified null or direction.", "",
        "| Model | Context | No-news signed mean [95% interval] | No-news mean absolute [95% interval] | Observed / planned | Repeated-news signed mean [95% interval] | Repeated-news mean absolute [95% interval] | Observed / planned |",
        "|---|---|---|---|---:|---|---|---:|",
    ])
    for model in MODEL_KEYS:
        for context in CONTEXTS:
            controls = report["models"][model]["control_updates"][context]
            null, repeat = controls["no_news"], controls["repeated_news"]
            n, r = null["signed_mean_pp"], repeat["signed_mean_pp"]
            lines.append(f"| {model} | {context} | {estimate_text(n)} | {estimate_text(null['mean_absolute_pp'])} | {n['n_observed']} / {n['n_planned']} | {estimate_text(r)} | {estimate_text(repeat['mean_absolute_pp'])} | {r['n_observed']} / {r['n_planned']} |")
    lines.extend(["", "Family resampling preserves all paired contexts, update arms, and repeats. This aggregate retains each model's analysis settings and full metric denominators in JSON. Models appear in the prescribed order, without ranking or selecting a winner.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-prefix", type=Path, help="Defaults to RESULTS_ROOT/development_summary")
    args = parser.parse_args(argv)
    try:
        report = aggregate(args.results_root)
        markdown = markdown_summary(report)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    prefix = args.output_prefix or args.results_root / "development_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    prefix.with_suffix(".md").write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
