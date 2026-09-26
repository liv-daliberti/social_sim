#!/usr/bin/env python3
"""Compute the Experiment 1 market-clustered movement intervals.

The June 17 consistency report is the frozen numerical authority for the
paper.  This script bootstraps whole markets from that report so that every
counterfactual packet and repeated initial-forecast update for a sampled
market remains together.  It also checks the reconstructed point estimates
and record counts against the report before writing an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "data/results/consistency_report_2026-06-17.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/results/clustered_movement_uncertainty.json"
DEFAULT_REPETITIONS = 20_000
DEFAULT_SEED = 20_260_822

DIRECTIONS = ("pro_H1", "anti_H1", "orthogonal")
EXCLUDED_MODELS = {
    "qwen2.5:7b": (
        "Archived magnitude records mix probability scales; the paper "
        "excludes this deployment from magnitude and sensitivity comparisons."
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direction(counterfactual: dict) -> tuple[str | None, bool]:
    """Return the intervention direction and whether it came from ``cf_id``."""
    direction = counterfactual.get("direction")
    if direction in DIRECTIONS:
        return direction, False

    cf_id = str(counterfactual.get("cf_id", ""))
    for candidate in DIRECTIONS:
        if f"_{candidate}_" in cf_id:
            return candidate, True
    return None, False


def _cluster_arrays(model_report: dict) -> tuple[dict, dict]:
    """Collect record sums and counts into one row per market."""
    markets = sorted(
        (row for row in model_report["per_market"] if row.get("n_updates", 0) > 0),
        key=lambda row: row["task_id"],
    )
    task_ids = [row["task_id"] for row in markets]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Market task IDs are not unique")
    if not markets:
        raise ValueError("Model report has no markets with valid updates")

    abs_sums = {direction: [] for direction in DIRECTIONS}
    signed_sums = {direction: [] for direction in DIRECTIONS}
    counts = {direction: [] for direction in DIRECTIONS}
    recovered_records = 0

    for market in markets:
        values = {direction: [] for direction in DIRECTIONS}
        for counterfactual in market.get("cf_results", []):
            direction, recovered = _direction(counterfactual)
            valid = [
                float(run["delta_yes_prob"])
                for run in counterfactual.get("runs", [])
                if run.get("delta_yes_prob") is not None
            ]
            if valid and direction is None:
                raise ValueError(
                    "Cannot recover direction for "
                    f"{market['task_id']} / {counterfactual.get('cf_id')}"
                )
            if direction is None:
                continue
            values[direction].extend(valid)
            if recovered:
                recovered_records += len(valid)

        for direction in DIRECTIONS:
            array = np.asarray(values[direction], dtype=float)
            abs_sums[direction].append(float(np.abs(array).sum()))
            signed_sums[direction].append(float(array.sum()))
            counts[direction].append(int(array.size))

    arrays = {}
    for name, source in (
        ("abs_sums", abs_sums),
        ("signed_sums", signed_sums),
        ("counts", counts),
    ):
        arrays[name] = {
            direction: np.asarray(source[direction], dtype=float)
            for direction in DIRECTIONS
        }

    metadata = {
        "market_ids": task_ids,
        "n_markets": len(markets),
        "direction_recovered_from_cf_id_records": recovered_records,
    }
    return arrays, metadata


def _estimate(sum_values: np.ndarray, counts: np.ndarray) -> float:
    denominator = float(counts.sum())
    if denominator <= 0:
        raise ValueError("Estimand has no valid records")
    return float(sum_values.sum() / denominator)


def _bootstrap_mean(
    sum_values: np.ndarray,
    counts: np.ndarray,
    sampled_indices: np.ndarray,
) -> np.ndarray:
    numerators = sum_values[sampled_indices].sum(axis=1)
    denominators = counts[sampled_indices].sum(axis=1)
    if np.any(denominators <= 0):
        raise ValueError("A bootstrap replicate contains no valid records")
    return numerators / denominators


def _summary(estimate: float, draws: np.ndarray) -> dict:
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def analyze_model(model_report: dict, repetitions: int, seed: int) -> dict:
    """Analyze one model and validate it against its frozen summaries."""
    arrays, metadata = _cluster_arrays(model_report)
    n_markets = metadata["n_markets"]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_markets, size=(repetitions, n_markets))

    signed = {}
    absolute = {}
    signed_draws = {}
    absolute_draws = {}
    for direction in DIRECTIONS:
        point_signed = _estimate(
            arrays["signed_sums"][direction], arrays["counts"][direction]
        )
        point_absolute = _estimate(
            arrays["abs_sums"][direction], arrays["counts"][direction]
        )
        signed_draws[direction] = _bootstrap_mean(
            arrays["signed_sums"][direction],
            arrays["counts"][direction],
            sampled,
        )
        absolute_draws[direction] = _bootstrap_mean(
            arrays["abs_sums"][direction],
            arrays["counts"][direction],
            sampled,
        )
        signed[direction] = _summary(point_signed, signed_draws[direction])
        absolute[direction] = _summary(point_absolute, absolute_draws[direction])
        record_count = int(arrays["counts"][direction].sum())
        signed[direction]["n_records"] = record_count
        absolute[direction]["n_records"] = record_count

    directional_sums = arrays["abs_sums"]["pro_H1"] + arrays["abs_sums"]["anti_H1"]
    directional_counts = arrays["counts"]["pro_H1"] + arrays["counts"]["anti_H1"]
    point_directional = _estimate(directional_sums, directional_counts)
    draws_directional = _bootstrap_mean(directional_sums, directional_counts, sampled)
    directional = _summary(point_directional, draws_directional)
    directional["n_records"] = int(directional_counts.sum())

    point_orthogonal = absolute["orthogonal"]["estimate"]
    point_sensitivity = point_directional / point_orthogonal
    sensitivity_draws = draws_directional / absolute_draws["orthogonal"]
    sensitivity = _summary(point_sensitivity, sensitivity_draws)

    # The frozen report rounds its movement summaries. These tolerances are
    # just wider than that rounding error, while still catching scale, filter,
    # or direction mistakes.
    anchoring = model_report["anchoring"]
    reported_directional = (
        anchoring["pro_H1"]["mean_abs_delta"] * anchoring["pro_H1"]["n"]
        + anchoring["anti_H1"]["mean_abs_delta"] * anchoring["anti_H1"]["n"]
    ) / (anchoring["pro_H1"]["n"] + anchoring["anti_H1"]["n"])
    if abs(point_directional - reported_directional) > 0.00006:
        raise ValueError(
            f"Directional estimate mismatch: {point_directional} vs "
            f"{reported_directional}"
        )
    if abs(point_orthogonal - anchoring["orthogonal"]["mean_abs_delta"]) > 0.00006:
        raise ValueError(
            f"Orthogonal estimate mismatch: {point_orthogonal} vs "
            f"{anchoring['orthogonal']['mean_abs_delta']}"
        )
    if abs(point_sensitivity - anchoring["sensitivity_ratio"]) > 0.0006:
        raise ValueError(
            f"Sensitivity mismatch: {point_sensitivity} vs "
            f"{anchoring['sensitivity_ratio']}"
        )
    for direction in DIRECTIONS:
        expected_n = anchoring[direction]["n"]
        observed_n = signed[direction]["n_records"]
        if observed_n != expected_n:
            raise ValueError(
                f"{direction} count mismatch: {observed_n} vs {expected_n}"
            )

    return {
        **metadata,
        "signed_revision": signed,
        "absolute_revision": {**absolute, "directional": directional},
        "sensitivity_ratio": sensitivity,
    }


def _source_label(source: Path) -> str:
    try:
        return str(source.resolve().relative_to(EXPERIMENT_ROOT))
    except ValueError:
        return str(source.resolve())


def build_report(source: Path, repetitions: int, seed: int) -> dict:
    """Build the complete clustered-bootstrap report from a frozen report."""
    source = source.resolve()
    frozen = json.loads(source.read_text(encoding="utf-8"))
    results = {}
    for model in frozen["models"]:
        if model in EXCLUDED_MODELS:
            continue
        results[model] = analyze_model(
            frozen["per_model"][model], repetitions=repetitions, seed=seed
        )

    return {
        "schema_version": 1,
        "source_report": _source_label(source),
        "source_report_sha256": _sha256(source),
        "bootstrap": {
            "cluster_unit": "market task_id",
            "repetitions": repetitions,
            "rng": "numpy.random.Generator(PCG64)",
            "seed": seed,
            "interval": "percentile 95% (2.5th, 97.5th percentiles)",
            "resampling": (
                "Sample markets with replacement; retain every packet and "
                "repeated initial-run update within each selected market."
            ),
            "estimand": (
                "Record-pooled mean within condition; sensitivity is the "
                "pooled directional absolute mean divided by the pooled "
                "orthogonal absolute mean, recomputed within each bootstrap "
                "replicate."
            ),
        },
        "excluded_models": EXCLUDED_MODELS,
        "per_model": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1_000:
        raise SystemExit("--repetitions must be at least 1000")
    if not args.input.is_file():
        raise SystemExit(f"Input report does not exist: {args.input}")

    report = build_report(args.input, args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
