#!/usr/bin/env python3
"""Fit train-only calibration baselines and score frozen Experiment 3B splits."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any



from family_clusters import connected_index_groups
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "exp3b_registered"
DEFAULT_REPORT = ROOT / "reports" / "exp3b_baselines.json"
EPS = 1e-6


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clamp(probability: float) -> float:
    return max(EPS, min(1.0 - EPS, float(probability)))


def logit(probability: float) -> float:
    p = clamp(probability)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fit_platt(rows: list[dict[str, Any]], l2: float = 1e-4) -> tuple[float, float]:
    """Fit sigmoid(intercept + slope * market_logit) by damped Newton steps."""
    intercept, slope = 0.0, 1.0
    xs = [logit(float(row["market_yes_prob"])) for row in rows]
    ys = [float(bool(row["settlement_yes"])) for row in rows]
    for _ in range(100):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            q = sigmoid(intercept + slope * x)
            residual = q - y
            weight = max(q * (1.0 - q), 1e-9)
            g0 += residual
            g1 += residual * x
            h00 += weight
            h01 += weight * x
            h11 += weight * x * x
        g1 += l2 * slope
        h11 += l2
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-12:
            raise ArithmeticError("singular Platt calibration Hessian")
        delta0 = (h11 * g0 - h01 * g1) / determinant
        delta1 = (-h01 * g0 + h00 * g1) / determinant
        intercept -= delta0
        slope -= delta1
        if max(abs(delta0), abs(delta1)) < 1e-10:
            break
    return intercept, slope


def metric_summary(labels: list[float], predictions: list[float]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be nonempty and aligned")
    probabilities = [clamp(value) for value in predictions]
    n = len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n
    log_loss = -sum(
        y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
        for p, y in zip(probabilities, labels)
    ) / n
    accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, labels)) / n
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10.0, (bin_index + 1) / 10.0
        members = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if low <= p < high or (bin_index == 9 and p == 1.0)
        ]
        if members:
            mean_p = sum(item[0] for item in members) / len(members)
            mean_y = sum(item[1] for item in members) / len(members)
            ece += len(members) / n * abs(mean_p - mean_y)
    return {
        "n": n,
        "brier": brier,
        "log_loss_nats": log_loss,
        "accuracy_at_0.5": accuracy,
        "ece_10_bin": ece,
        "mean_forecast": sum(probabilities) / n,
        "yes_rate": sum(labels) / n,
    }


def cluster_bootstrap_brier_delta(
    rows: list[dict[str, Any]],
    first: list[float],
    second: list[float],
    repetitions: int = 5_000,
) -> dict[str, float]:
    """Family-cluster bootstrap for Brier(first) minus Brier(second)."""
    groups = connected_index_groups(rows)
    rng = random.Random(30_032_026)
    deltas = []
    for _ in range(repetitions):
        sampled = [rng.choice(groups) for _ in groups]
        numerator = 0.0
        count = 0
        for group in sampled:
            for index in group:
                y = float(bool(rows[index]["settlement_yes"]))
                numerator += (clamp(first[index]) - y) ** 2
                numerator -= (clamp(second[index]) - y) ** 2
                count += 1
        deltas.append(numerator / count)
    deltas.sort()
    return {
        "estimate": sum(
            (clamp(p1) - float(bool(row["settlement_yes"]))) ** 2
            - (clamp(p2) - float(bool(row["settlement_yes"]))) ** 2
            for row, p1, p2 in zip(rows, first, second)
        ) / len(rows),
        "family_cluster_count": len(groups),
        "ci95_low": deltas[int(0.025 * repetitions)],
        "ci95_high": deltas[int(0.975 * repetitions)],
        "bootstrap_repetitions": repetitions,
    }


def evaluate(data: Path) -> dict[str, Any]:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("protocol_version") != "exp3b_registered_v1":
        raise AssertionError("dataset manifest is not the passed registered protocol")
    splits = {name: read_jsonl(data / f"{name}.tasks.jsonl") for name in ("train", "dev", "test")}
    train_yes_rate = sum(bool(row["settlement_yes"]) for row in splits["train"]) / len(splits["train"])
    intercept, slope = fit_platt(splits["train"])
    report: dict[str, Any] = {
        "protocol_version": "exp3b_registered_v1",
        "fit_on": "train only",
        "platt_parameters": {"intercept": intercept, "market_logit_slope": slope, "l2": 1e-4},
        "train_yes_rate": train_yes_rate,
        "splits": {},
    }
    for name, rows in splits.items():
        labels = [float(bool(row["settlement_yes"])) for row in rows]
        market = [float(row["market_yes_prob"]) for row in rows]
        calibrated = [sigmoid(intercept + slope * logit(value)) for value in market]
        climatology = [train_yes_rate] * len(rows)
        report["splits"][name] = {
            "market": metric_summary(labels, market),
            "train_climatology": metric_summary(labels, climatology),
            "platt_market_train_only": metric_summary(labels, calibrated),
            "platt_minus_market_brier": cluster_bootstrap_brier_delta(rows, calibrated, market),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
