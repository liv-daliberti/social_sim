#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_baselines as baselines  # noqa: E402


class BaselineTest(unittest.TestCase):
    def test_platt_fit_improves_deliberately_miscalibrated_prices(self) -> None:
        rows = []
        for index in range(100):
            label = index % 2 == 0
            # Correct direction, deliberately too close to 0.5.
            probability = 0.55 if label else 0.45
            rows.append({"market_yes_prob": probability, "settlement_yes": label})
        intercept, slope = baselines.fit_platt(rows)
        raw = [row["market_yes_prob"] for row in rows]
        fitted = [
            baselines.sigmoid(intercept + slope * baselines.logit(value))
            for value in raw
        ]
        labels = [float(row["settlement_yes"]) for row in rows]
        self.assertLess(
            baselines.metric_summary(labels, fitted)["brier"],
            baselines.metric_summary(labels, raw)["brier"],
        )

    def test_cluster_bootstrap_sign(self) -> None:
        rows = [
            {"event_id": str(index // 2), "settlement_yes": bool(index % 2)}
            for index in range(20)
        ]
        labels = [float(row["settlement_yes"]) for row in rows]
        perfect = [0.99 if label else 0.01 for label in labels]
        uninformative = [0.5] * len(rows)
        delta = baselines.cluster_bootstrap_brier_delta(
            rows, perfect, uninformative, repetitions=200
        )
        self.assertLess(delta["estimate"], 0.0)
        self.assertLess(delta["ci95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
