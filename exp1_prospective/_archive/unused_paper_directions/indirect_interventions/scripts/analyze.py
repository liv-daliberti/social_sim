#!/usr/bin/env python3
"""Analyze frozen indirect updates with registered clustered endpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INSTRUMENT = ROOT / "data/frozen/instrument.jsonl"
DEFAULT_MANIFEST = ROOT / "data/frozen/manifest.json"
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_REPORT = ROOT / "reports/target_results.json"
DEFAULT_TABLE = ROOT / "reports/target_results.tex"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def condition_correct(condition: str, delta: float | None, tau: float) -> bool:
    if delta is None:
        return False
    if condition in {"positive", "direct_pro"}:
        return delta >= tau
    if condition in {"negative", "direct_anti"}:
        return delta <= -tau
    if condition in {"broken", "orthogonal", "literal_null"}:
        return abs(delta) < tau
    raise ValueError(f"condition {condition!r} has no registered directional score")


def cluster_bootstrap(
    values: dict[str, float],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    market_ids = sorted(values)
    observed = np.array([values[market_id] for market_id in market_ids], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observed), size=(replicates, len(observed)))
    return observed[indices].mean(axis=1)


def interval(draws: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1 - alpha)),
    ]


def market_metrics(rows: list[dict[str, Any]], tau: float) -> dict[str, dict[str, float]]:
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_market[row["task_id"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for market_id, market_rows in by_market.items():
        by_candidate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in market_rows:
            by_candidate[row["candidate_id"]][row["condition"]] = row
        families = [
            conditions for conditions in by_candidate.values()
            if {"positive", "negative", "broken", "masked"} <= set(conditions)
        ]
        reversal = []
        label_scores: dict[str, list[float]] = defaultdict(list)
        complete_triplets = []
        masked_movement = []
        bridge_movement = []
        for family in families:
            positive = family["positive"].get("delta_yes_prob")
            negative = family["negative"].get("delta_yes_prob")
            # Invalid pairs remain in the denominator as a conservative zero contrast.
            reversal.append(
                float(positive) - float(negative)
                if positive is not None and negative is not None
                else 0.0
            )
            triplet = []
            for condition in ("positive", "negative", "broken"):
                correct = float(condition_correct(
                    condition,
                    family[condition].get("delta_yes_prob"),
                    tau,
                ))
                label_scores[condition].append(correct)
                triplet.append(bool(correct))
                delta = family[condition].get("delta_yes_prob")
                bridge_movement.append(abs(float(delta)) if delta is not None else 0.0)
            complete_triplets.append(float(all(triplet)))
            masked = family["masked"].get("delta_yes_prob")
            masked_movement.append(abs(float(masked)) if masked is not None else 0.0)

        controls = {
            condition: next(
                (
                    row for row in market_rows
                    if row["condition"] == condition
                ),
                None,
            )
            for condition in ("direct_pro", "direct_anti", "orthogonal", "literal_null")
        }
        control_correct = [
            float(condition_correct(condition, row.get("delta_yes_prob"), tau))
            for condition, row in controls.items()
            if row is not None
        ]
        literal = controls["literal_null"]
        literal_drift = (
            abs(float(literal["delta_yes_prob"]))
            if literal is not None and literal.get("delta_yes_prob") is not None
            else 0.0
        )
        macro_triplet = np.mean([
            np.mean(label_scores[label]) if label_scores[label] else 0.0
            for label in ("positive", "negative", "broken")
        ])
        direct_directional = [
            float(condition_correct(condition, controls[condition].get("delta_yes_prob"), tau))
            for condition in ("direct_pro", "direct_anti")
            if controls[condition] is not None
        ]
        indirect_directional = label_scores["positive"] + label_scores["negative"]
        result[market_id] = {
            "reversal_contrast": float(np.mean(reversal)) if reversal else 0.0,
            "triplet_accuracy": float(macro_triplet),
            "complete_triplet": float(np.mean(complete_triplets)) if complete_triplets else 0.0,
            "positive_accuracy": float(np.mean(label_scores["positive"])) if label_scores["positive"] else 0.0,
            "negative_accuracy": float(np.mean(label_scores["negative"])) if label_scores["negative"] else 0.0,
            "broken_specificity": float(np.mean(label_scores["broken"])) if label_scores["broken"] else 0.0,
            "direct_control_accuracy": float(np.mean(control_correct)) if control_correct else 0.0,
            "direct_to_indirect_drop": (
                float(np.mean(direct_directional) - np.mean(indirect_directional))
                if direct_directional and indirect_directional else 0.0
            ),
            "bridge_minus_masked_movement": (
                float(np.mean(bridge_movement) - np.mean(masked_movement))
                if bridge_movement and masked_movement else 0.0
            ),
            "literal_null_drift": literal_drift,
        }
    return result


def summarize_model(
    model: str,
    rows: list[dict[str, Any]],
    *,
    expected_packets: int,
    tau: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics = market_metrics(rows, tau)
    names = (
        "reversal_contrast",
        "triplet_accuracy",
        "complete_triplet",
        "positive_accuracy",
        "negative_accuracy",
        "broken_specificity",
        "direct_control_accuracy",
        "direct_to_indirect_drop",
        "bridge_minus_masked_movement",
        "literal_null_drift",
    )
    draws = {
        name: cluster_bootstrap(
            {market: values[name] for market, values in metrics.items()},
            replicates=replicates,
            seed=seed + index,
        )
        for index, name in enumerate(names)
    }
    estimates = {
        name: {
            "estimate": float(np.mean([values[name] for values in metrics.values()])),
            "ci95": interval(draws[name]),
        }
        for name in names
    }
    parse_count = sum(bool(row.get("parse_valid")) for row in rows)
    parse_coverage = parse_count / expected_packets if expected_packets else 0
    summary = {
        "model": model,
        "records": len(rows),
        "expected_records": expected_packets,
        "parse_valid": parse_count,
        "parse_coverage": parse_coverage,
        "market_count": len(metrics),
        "metrics": estimates,
    }
    return summary, draws


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, model in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[model])
        running = max(running, value)
        adjusted[model] = running
    return adjusted


def render_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & $C_m$ & Triplet acc. & Complete & Parse \\",
        r"\midrule",
    ]
    for row in summaries:
        metrics = row["metrics"]
        lines.append(
            f"{row['model']} & {metrics['reversal_contrast']['estimate']:.3f} & "
            f"{metrics['triplet_accuracy']['estimate']:.3f} & "
            f"{metrics['complete_triplet']['estimate']:.3f} & "
            f"{row['parse_coverage']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tau = float(manifest["tau"])
    expected_packets = int(manifest["packet_count"])
    rows = []
    for path in sorted(args.runs.glob("*/responses.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit(f"no target response files under {args.runs}")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("instrument_sha256") != manifest["instrument_sha256"]:
            raise SystemExit("run contains a response from a different instrument hash")
        by_model[row["model"]].append(row)

    summaries, reversal_draws = [], {}
    for index, (model, model_rows) in enumerate(sorted(by_model.items())):
        summary, draws = summarize_model(
            model,
            model_rows,
            expected_packets=expected_packets,
            tau=tau,
            replicates=args.bootstrap,
            seed=args.seed + 100 * index,
        )
        summaries.append(summary)
        reversal_draws[model] = draws["reversal_contrast"]
    raw_p = {
        model: float((np.sum(draws <= 0) + 1) / (len(draws) + 1))
        for model, draws in reversal_draws.items()
    }
    adjusted_p = holm_adjust(raw_p)
    ordered = sorted(raw_p, key=raw_p.get)
    m = len(ordered)
    for rank, model in enumerate(ordered):
        alpha = 0.05 / (m - rank)
        row = next(item for item in summaries if item["model"] == model)
        row["reversal_test"] = {
            "bootstrap_one_sided_p": raw_p[model],
            "holm_adjusted_p": adjusted_p[model],
            "holm_adjusted_lower_bound": float(np.quantile(reversal_draws[model], alpha)),
            "pass": (
                float(np.quantile(reversal_draws[model], alpha)) > 0
                and row["parse_coverage"] >= 0.95
            ),
        }
    report = {
        "protocol_version": "indirect-core-v1",
        "instrument_sha256": manifest["instrument_sha256"],
        "tau": tau,
        "bootstrap_replicates": args.bootstrap,
        "cluster": "market",
        "models": summaries,
        "note": (
            "Claim gates involving paired shortcut-baseline superiority remain "
            "pending until the frozen baseline report is supplied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.table.write_text(render_table(summaries), encoding="utf-8")
    print(f"analyzed {len(summaries)} models -> {args.output}")


if __name__ == "__main__":
    main()
