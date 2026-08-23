#!/usr/bin/env python3
"""Analyze the paired Experiment 1 reapplication for base and 3B adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_NAMES = ("base", "seed_42", "seed_43", "seed_44")
DIRECTIONAL = ("pro_H1", "anti_H1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def mean(values: Iterable[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def mean_se(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    average = sum(values) / len(values)
    if len(values) == 1:
        return average, None
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average, math.sqrt(variance / len(values))


def record_metrics(record: dict[str, Any]) -> dict[str, float | int | None]:
    direction = record.get("direction")
    delta_yes = record.get("delta_yes_prob")
    delta_h1 = record.get("delta_h1_prob")
    updated_yes = record.get("updated_yes_prob")
    updated_h1 = record.get("updated_h1_prob")

    ehc = None
    if direction in DIRECTIONAL and delta_h1 is not None and abs(delta_h1) >= 0.03:
        ehc = int((direction == "pro_H1" and delta_h1 > 0) or (direction == "anti_H1" and delta_h1 < 0))
    hfc = None
    if direction in DIRECTIONAL and delta_yes is not None and abs(delta_yes) >= 0.03:
        hfc = int((direction == "pro_H1" and delta_yes > 0) or (direction == "anti_H1" and delta_yes < 0))
    ics = None
    if updated_yes is not None and updated_h1 is not None:
        ics = int(abs(updated_yes - updated_h1) < 0.02)
    return {
        "EHC": ehc,
        "HFC": hfc,
        "ICS": ics,
        "abs_delta": abs(delta_yes) if delta_yes is not None else None,
    }


def market_rates(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = record_metrics(row)[metric]
        if value is not None:
            grouped[str(row["task_id"])].append(float(value))
    return {task_id: sum(values) / len(values) for task_id, values in grouped.items()}


def magnitudes_by_market(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = record_metrics(row)["abs_delta"]
        if value is not None:
            grouped[str(row["task_id"])][str(row["direction"])].append(float(value))
    return {task: dict(directions) for task, directions in grouped.items()}


def sensitivity_from_markets(
    grouped: dict[str, dict[str, list[float]]], task_ids: list[str]
) -> float | None:
    directional = []
    orthogonal = []
    for task_id in task_ids:
        values = grouped.get(task_id, {})
        for direction in DIRECTIONAL:
            directional.extend(values.get(direction, []))
        orthogonal.extend(values.get("orthogonal", []))
    directional_mean = mean(directional)
    orthogonal_mean = mean(orthogonal)
    if directional_mean is None or orthogonal_mean in (None, 0.0):
        return None
    return directional_mean / orthogonal_mean


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = sorted({str(row["task_id"]) for row in rows})
    output: dict[str, Any] = {
        "n_markets": len(task_ids),
        "n_packets": len(rows),
        "parse_coverage": sum(
            row.get("updated_yes_prob") is not None and row.get("updated_h1_prob") is not None
            for row in rows
        )
        / len(rows),
    }
    for metric in ("EHC", "HFC", "ICS"):
        rates = market_rates(rows, metric)
        average, standard_error = mean_se(rates.values())
        output[metric] = {
            "rate": average,
            "se": standard_error,
            "n_markets": len(rates),
            "n_records": sum(record_metrics(row)[metric] is not None for row in rows),
        }

    direction_rows = {}
    for direction in ("pro_H1", "anti_H1", "orthogonal"):
        values = [
            float(record_metrics(row)["abs_delta"])
            for row in rows
            if row.get("direction") == direction and record_metrics(row)["abs_delta"] is not None
        ]
        average, standard_error = mean_se(values)
        direction_rows[direction] = {
            "mean_abs_delta": average,
            "se": standard_error,
            "n": len(values),
        }
    output["revision_magnitude"] = direction_rows
    output["sensitivity"] = sensitivity_from_markets(magnitudes_by_market(rows), task_ids)
    return output


def percentile_interval(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"ci95_low": None, "ci95_high": None}
    samples.sort()
    count = len(samples)
    return {
        "ci95_low": samples[int(0.025 * count)],
        "ci95_high": samples[int(0.975 * count)],
    }


def paired_bootstrap(
    rows_by_name: dict[str, list[dict[str, Any]]], repetitions: int
) -> dict[str, dict[str, float]]:
    market_sets = [
        {str(row["task_id"]) for row in rows}
        for rows in rows_by_name.values()
    ]
    if not market_sets or any(market_set != market_sets[0] for market_set in market_sets[1:]):
        raise AssertionError("model outputs do not cover identical market sets")
    task_ids = sorted(market_sets[0])
    rng = random.Random(30_032_026)
    rate_maps = {
        name: {metric: market_rates(rows, metric) for metric in ("EHC", "HFC", "ICS")}
        for name, rows in rows_by_name.items()
    }
    magnitude_maps = {
        name: magnitudes_by_market(rows) for name, rows in rows_by_name.items()
    }
    trained = ("seed_42", "seed_43", "seed_44")
    results: dict[str, dict[str, float]] = {}

    for metric in ("EHC", "HFC", "ICS"):
        samples = []
        for _ in range(repetitions):
            selected = [rng.choice(task_ids) for _ in task_ids]
            base_values = [rate_maps["base"][metric].get(task_id) for task_id in selected]
            trained_values = [
                mean(
                    rate_maps[name][metric].get(task_id)
                    for name in trained
                    if rate_maps[name][metric].get(task_id) is not None
                )
                for task_id in selected
            ]
            paired = [
                (trained_value, base_value)
                for trained_value, base_value in zip(trained_values, base_values)
                if trained_value is not None and base_value is not None
            ]
            if paired:
                samples.append(mean(a - b for a, b in paired))
        estimate_pairs = []
        for task_id in task_ids:
            trained_value = mean(
                rate_maps[name][metric].get(task_id)
                for name in trained
                if rate_maps[name][metric].get(task_id) is not None
            )
            base_value = rate_maps["base"][metric].get(task_id)
            if trained_value is not None and base_value is not None:
                estimate_pairs.append(trained_value - base_value)
        results[f"trained_mean_minus_base_{metric}"] = {
            "estimate": mean(estimate_pairs),
            **percentile_interval(samples),
            "n_markets": len(estimate_pairs),
            "bootstrap_repetitions": repetitions,
        }

    sensitivity_samples = []
    for _ in range(repetitions):
        selected = [rng.choice(task_ids) for _ in task_ids]
        base_value = sensitivity_from_markets(magnitude_maps["base"], selected)
        trained_value = mean(
            value
            for name in trained
            if (value := sensitivity_from_markets(magnitude_maps[name], selected)) is not None
        )
        if base_value is not None and trained_value is not None:
            sensitivity_samples.append(trained_value - base_value)
    base_sensitivity = sensitivity_from_markets(magnitude_maps["base"], task_ids)
    trained_sensitivity = mean(
        sensitivity_from_markets(magnitude_maps[name], task_ids) for name in trained
    )
    sensitivity_difference = (
        trained_sensitivity - base_sensitivity
        if trained_sensitivity is not None and base_sensitivity is not None
        else None
    )
    results["trained_mean_minus_base_sensitivity"] = {
        "estimate": sensitivity_difference,
        **percentile_interval(sensitivity_samples),
        "n_markets": len(task_ids),
        "bootstrap_repetitions": repetitions,
    }
    return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(paths: list[Path], repetitions: int) -> dict[str, Any]:
    rows_by_name = {}
    inputs = {}
    key_set = None
    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            raise AssertionError(f"empty evaluation output: {path}")
        name = str(rows[0].get("evaluation_name"))
        if name in rows_by_name:
            raise AssertionError(f"duplicate evaluation name: {name}")
        if any(row.get("evaluation_name") != name for row in rows):
            raise AssertionError(f"mixed evaluation names in {path}")
        keys = {(str(row["task_id"]), str(row["cf_id"])) for row in rows}
        if len(keys) != len(rows):
            raise AssertionError(f"duplicate task/packet rows in {path}")
        if key_set is None:
            key_set = keys
        elif keys != key_set:
            raise AssertionError(f"instrument mismatch in {path}")
        rows_by_name[name] = rows
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}

    if tuple(sorted(rows_by_name)) != tuple(sorted(EXPECTED_NAMES)):
        raise AssertionError(f"expected {EXPECTED_NAMES}, got {sorted(rows_by_name)}")
    summaries = {name: summarize_model(rows_by_name[name]) for name in EXPECTED_NAMES}
    trained_mean = {}
    for metric in ("EHC", "HFC", "ICS"):
        trained_mean[metric] = mean(summaries[name][metric]["rate"] for name in EXPECTED_NAMES[1:])
    trained_mean["sensitivity"] = mean(
        summaries[name]["sensitivity"] for name in EXPECTED_NAMES[1:]
    )
    trained_mean["parse_coverage"] = mean(
        summaries[name]["parse_coverage"] for name in EXPECTED_NAMES[1:]
    )
    return {
        "protocol_version": "exp3b_exp1_reapplication_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "models": summaries,
        "trained_mean": trained_mean,
        "comparisons": paired_bootstrap(rows_by_name, repetitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    summary = analyze(args.inputs, args.bootstrap_repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
