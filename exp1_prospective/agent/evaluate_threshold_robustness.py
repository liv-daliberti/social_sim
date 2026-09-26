#!/usr/bin/env python3
"""Threshold and unconditional robustness analysis for Experiment 1.

This is deliberately a separate, read-only reanalysis of the frozen Stage 3
records. It uses the same cross-file de-duplication rule as
evaluate_consistency.py but normalizes both the headline probability and the
duplicated H1 posterior before applying percentage-point thresholds.

Outputs:
  data/results/threshold_robustness.json
  paper/tables/exp1_threshold_robustness.tex
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evaluate_consistency import (
    _load_all_updated_forecasts,
    _norm01,
    _norm_tid,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
UPDATE_DIR = ROOT / "data" / "updated_forecasts"
OUT_JSON = ROOT / "data" / "results" / "threshold_robustness.json"
OUT_TEX = REPO_ROOT / "paper" / "tables" / "exp1_threshold_robustness.tex"

THRESHOLDS = (0.00, 0.01, 0.03, 0.05)
RELEVANT_DIRECTIONS = {"pro_H1": 1, "anti_H1": -1}

# Hosted systems first; open-weight rows in descending parameter count.
MODEL_ORDER = (
    "claude-opus-4-8",
    "gpt-5.4",
    "DeepSeek-V4-Pro",
    "qwen2.5:72b",
    "llama3.3:70b",
    "llama3.1:70b",
    "qwen2.5:32b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "qwen2.5:7b",
)

MODEL_LABELS = {
    "claude-opus-4-8": "Claude Opus~4.8",
    "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro",
    "llama3.3:70b": "Llama-3.3-70B",
    "llama3.1:70b": "Llama-3.1-70B",
    "qwen2.5:72b": "Qwen2.5-72B",
    "qwen2.5:32b": "Qwen2.5-32B",
    "qwen2.5:14b": "Qwen2.5-14B",
    "qwen2.5:7b": "Qwen2.5-7B",
    "llama3.1:8b": "Llama-3.1-8B",
}


def _finite_probability(value) -> float | None:
    """Normalize the repository's documented [0,1]/[0,100] outputs."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    value = float(_norm01(value))
    return value if 0 <= value <= 1 else None


def _h1_posterior(record: dict) -> float | None:
    structured = record.get("updated_structured_forecast") or {}
    hypotheses = structured.get("hypotheses") or []
    for hypothesis in hypotheses:
        if isinstance(hypothesis, dict) and hypothesis.get("id") == "H1":
            return _finite_probability(hypothesis.get("posterior_probability"))
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _threshold_summary(
    by_market: dict[str, list[tuple[float, int]]],
    threshold: float,
) -> dict:
    """Return the original market-macro estimand and transparent coverage."""
    market_rates = []
    eligible_records = 0
    correct_records = 0
    for observations in by_market.values():
        eligible = [
            (delta, direction)
            for delta, direction in observations
            if abs(delta) + 1e-12 >= threshold
        ]
        if not eligible:
            continue
        correct = sum(direction * delta > 0 for delta, direction in eligible)
        market_rates.append(correct / len(eligible))
        eligible_records += len(eligible)
        correct_records += correct

    valid_records = sum(len(rows) for rows in by_market.values())
    return {
        "threshold_pp": int(round(100 * threshold)),
        "market_macro_directional_correctness": _mean(market_rates),
        "market_macro_standard_error": _standard_error(market_rates),
        "eligible_records": eligible_records,
        "eligible_fraction": eligible_records / valid_records if valid_records else None,
        "correct_records": correct_records,
        "markets_with_eligible_records": len(market_rates),
    }


def _metric_summary(observations: list[dict], delta_key: str) -> dict:
    by_market: dict[str, list[tuple[float, int]]] = defaultdict(list)
    invalid_records = 0
    for observation in observations:
        delta = observation.get(delta_key)
        if delta is None:
            invalid_records += 1
            continue
        by_market[observation["market"]].append((delta, observation["direction_sign"]))

    valid_records = sum(len(rows) for rows in by_market.values())
    below_3pp = sum(
        abs(delta) < 0.03 - 1e-12
        for rows in by_market.values()
        for delta, _ in rows
    )
    return {
        "valid_relevant_records": valid_records,
        "invalid_relevant_records": invalid_records,
        "markets_with_valid_records": len(by_market),
        "below_3pp_records": below_3pp,
        "below_3pp_fraction": below_3pp / valid_records if valid_records else None,
        "thresholds": {
            str(int(round(100 * threshold))): _threshold_summary(by_market, threshold)
            for threshold in THRESHOLDS
        },
    }


def _analyze_model(records: list[dict]) -> dict:
    observations = []
    schema_gaps = []
    raw_scale_mismatches = 0

    for record in records:
        direction = record.get("direction")
        if direction not in RELEVANT_DIRECTIONS:
            continue

        initial = _finite_probability(record.get("initial_yes_prob"))
        headline = _finite_probability(record.get("updated_yes_prob"))
        h1 = _h1_posterior(record)

        raw_h1 = None
        structured = record.get("updated_structured_forecast") or {}
        for hypothesis in structured.get("hypotheses") or []:
            if isinstance(hypothesis, dict) and hypothesis.get("id") == "H1":
                raw_h1 = hypothesis.get("posterior_probability")
                break
        if (
            isinstance(raw_h1, (int, float))
            and raw_h1 > 1.5
            and isinstance(record.get("updated_yes_prob"), (int, float))
            and record["updated_yes_prob"] <= 1.5
        ):
            raw_scale_mismatches += 1

        h1_delta = h1 - initial if h1 is not None and initial is not None else None
        headline_delta = (
            headline - initial
            if headline is not None and initial is not None
            else None
        )
        if h1 is not None and headline is not None:
            schema_gaps.append(abs(headline - h1))

        observations.append(
            {
                "market": _norm_tid(record.get("task_id")) or record.get("task_id", ""),
                "direction_sign": RELEVANT_DIRECTIONS[direction],
                "h1_delta": h1_delta,
                "headline_delta": headline_delta,
            }
        )

    schema_n = len(schema_gaps)
    exact_schema = sum(gap <= 1e-12 for gap in schema_gaps)
    within_2pp = sum(gap < 0.02 for gap in schema_gaps)
    return {
        "relevant_records": len(observations),
        "EHC": _metric_summary(observations, "h1_delta"),
        "HFC": _metric_summary(observations, "headline_delta"),
        "schema_adherence": {
            "paired_valid_records": schema_n,
            "exact_after_normalization_records": exact_schema,
            "exact_after_normalization_fraction": (
                exact_schema / schema_n if schema_n else None
            ),
            "within_2pp_records": within_2pp,
            "within_2pp_fraction": within_2pp / schema_n if schema_n else None,
            "raw_h1_headline_scale_mismatch_records": raw_scale_mismatches,
        },
    }


def build_report() -> dict:
    index = _load_all_updated_forecasts()
    records_by_model: dict[str, list[dict]] = defaultdict(list)
    for (_, model), records in index.items():
        records_by_model[model].extend(records)

    unknown_models = sorted(set(records_by_model) - set(MODEL_ORDER))
    if unknown_models:
        raise RuntimeError(f"Unregistered models in frozen records: {unknown_models}")
    missing_models = sorted(set(MODEL_ORDER) - set(records_by_model))
    if missing_models:
        raise RuntimeError(f"Predeclared models missing frozen records: {missing_models}")

    source_files = sorted(UPDATE_DIR.glob("updated_*.jsonl"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "directions": ["pro_H1", "anti_H1"],
            "thresholds_pp": [0, 1, 3, 5],
            "threshold_rule": "eligible iff abs(delta) >= threshold",
            "correctness_rule": "expected_direction * delta > 0",
            "threshold_0_interpretation": (
                "unconditional over all valid relevant records; zero movement is incorrect"
            ),
            "aggregation": (
                "directional correctness is computed within market and averaged equally "
                "across markets; eligibility fractions are record-weighted"
            ),
            "normalization": (
                "initial yes, updated yes, and updated H1 are normalized independently "
                "from the documented [0,1]/[0,100] formats before deltas are computed"
            ),
            "deduplication": (
                "same rule as evaluate_consistency._load_all_updated_forecasts"
            ),
        },
        "source_files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_files
        ],
        "model_order": list(MODEL_ORDER),
        "per_model": {
            model: _analyze_model(records_by_model[model])
            for model in MODEL_ORDER
        },
    }


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _tex_table(report: dict) -> str:
    lines = [
        "% Generated by exp1_prospective/agent/evaluate_threshold_robustness.py",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{Directional correctness} & \\",
        r"\cmidrule(lr){4-7}",
        r"\textbf{Model} & \textbf{Metric} & \textbf{$n$ valid} &",
        r"\textbf{0 pp} & \textbf{1 pp} & \textbf{3 pp} & \textbf{5 pp} &",
        r"\textbf{Below 3 pp} \\",
        r"\midrule",
    ]

    frontier_break_after = "DeepSeek-V4-Pro"
    for model in MODEL_ORDER:
        result = report["per_model"][model]
        for row_index, metric in enumerate(("EHC", "HFC")):
            metric_result = result[metric]
            threshold_results = metric_result["thresholds"]
            label = MODEL_LABELS[model] if row_index == 0 else ""
            cells = [
                label,
                metric,
                f"{metric_result['valid_relevant_records']:,}",
                *[
                    _pct(
                        threshold_results[str(threshold)][
                            "market_macro_directional_correctness"
                        ]
                    )
                    for threshold in (0, 1, 3, 5)
                ],
                (
                    f"{_pct(metric_result['below_3pp_fraction'])}"
                    f" ({metric_result['below_3pp_records']:,})"
                ),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if model == frontier_break_after:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    rendered_table = _tex_table(report)
    OUT_TEX.write_text(rendered_table)

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")
    for model in MODEL_ORDER:
        result = report["per_model"][model]
        ehc0 = result["EHC"]["thresholds"]["0"][
            "market_macro_directional_correctness"
        ]
        hfc0 = result["HFC"]["thresholds"]["0"][
            "market_macro_directional_correctness"
        ]
        print(
            f"{model:20s} "
            f"EHC0={_pct(ehc0):>7s} HFC0={_pct(hfc0):>7s} "
            f"below3(E/H)={_pct(result['EHC']['below_3pp_fraction'])}/"
            f"{_pct(result['HFC']['below_3pp_fraction'])}"
        )


if __name__ == "__main__":
    main()
