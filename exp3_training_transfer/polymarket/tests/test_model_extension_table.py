#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_model_extension_table as roster  # noqa: E402


def fake_summary(protocol: str, model_key: str = "qwen3_8b") -> dict:
    models = {
        "base": {
            "brier": 0.20,
            "log_loss_nats": 0.60,
            "n": 1024,
            "parse_coverage": 1.0,
        }
    }
    for seed, brier in zip((42, 43, 44), (0.11, 0.12, 0.13)):
        models[f"seed_{seed}"] = {
            "brier": brier,
            "log_loss_nats": brier + 0.20,
            "n": 1024,
            "parse_coverage": 0.99,
        }
    comparison = {
        "estimate": -0.08,
        "ci95_low": -0.10,
        "ci95_high": -0.06,
        "bootstrap_repetitions": 5000,
        "family_clusters": 466,
    }
    return {
        "protocol_version": protocol,
        "frozen_parent_protocol": "exp3b_registered_v1",
        "test_was_used_during_training": False,
        "model": roster.MODEL_NAMES[model_key],
        "model_key": model_key,
        "decoding": {"temperature": 0.0, "max_tokens": 128},
        "adapter_paths": {
            str(seed): f"/tmp/train_j{100 + seed}/adapter" for seed in (42, 43, 44)
        },
        "models": models,
        "baselines": {
            "market": {
                "brier": 0.14,
                "log_loss_nats": 0.4,
                "n": 1024,
                "parse_coverage": 1.0,
            },
            "platt_market_train_only": {
                "brier": 0.13,
                "log_loss_nats": 0.39,
                "n": 1024,
                "parse_coverage": 1.0,
            },
        },
        "comparisons_brier": {
            "trained_seed_mean_minus_base": comparison,
            "trained_seed_mean_minus_market": {**comparison, "estimate": -0.02},
            "trained_seed_mean_minus_platt_market": {**comparison, "estimate": -0.01},
        },
    }


class ModelExtensionTableTest(unittest.TestCase):
    def test_row_and_latex_preserve_seed_level_results(self) -> None:
        summary = fake_summary("exp3b_model_extension_v1")
        row = roster.row_from_summary(summary, "qwen3_8b", "Qwen3-8B")
        self.assertAlmostEqual(row["trained_mean_brier"], 0.12)
        self.assertAlmostEqual(row["minimum_parse_coverage"], 0.99)
        latex = roster.render_latex([row])
        self.assertIn("Qwen3-8B", latex)
        self.assertIn(".1100", latex)
        self.assertIn("-.0800 [-.10000,-.06000]", latex)
        self.assertIn(r"$\Delta$ Platt", latex)
        self.assertIn("-.0100 [-.10000,-.06000]", latex)

    def test_raw_validator_requires_exact_frozen_task_universe(self) -> None:
        test_rows = []
        rows = []
        for index in range(1024):
            task_id = f"task-{index}"
            reference = {
                "task_id": task_id,
                "event_id": f"event-{index // 2}",
                "settlement_yes": bool(index % 2),
                "market_yes_prob": 0.25,
            }
            test_rows.append(reference)
            rows.append(
                {
                    **reference,
                    "forecasts": {name: 0.5 for name in roster.EXPECTED_OUTPUTS},
                    "responses": {name: "{}" for name in roster.EXPECTED_OUTPUTS},
                }
            )
        summary = fake_summary("exp3b_model_extension_v1")
        for metrics in summary["models"].values():
            metrics["brier"] = 0.25
            metrics["parse_coverage"] = 1.0
        roster.validate_raw_rows(rows, test_rows, summary, "qwen3_8b")
        with self.assertRaises(AssertionError):
            roster.validate_raw_rows(rows[:-1], test_rows, summary, "qwen3_8b")
        drifted = [dict(row) for row in rows]
        drifted[0] = {**drifted[0], "market_yes_prob": 0.75}
        with self.assertRaises(AssertionError):
            roster.validate_raw_rows(drifted, test_rows, summary, "qwen3_8b")

    def test_raw_validator_accepts_and_recomputes_registered_parse_failures(
        self,
    ) -> None:
        test_rows = []
        rows = []
        for index in range(1024):
            task_id = f"task-{index}"
            reference = {
                "task_id": task_id,
                "event_id": f"event-{index // 2}",
                "settlement_yes": False,
                "market_yes_prob": 0.25,
            }
            forecasts = {name: 0.0 for name in roster.EXPECTED_OUTPUTS}
            if index < 12:
                forecasts["base"] = None
            test_rows.append(reference)
            rows.append(
                {
                    **reference,
                    "forecasts": forecasts,
                    "responses": {name: "{}" for name in roster.EXPECTED_OUTPUTS},
                }
            )
        summary = fake_summary("exp3b_model_extension_v1")
        summary["models"]["base"]["parse_coverage"] = 1012 / 1024
        summary["models"]["base"]["brier"] = 12 / 1024
        for seed in roster.SEEDS:
            summary["models"][f"seed_{seed}"]["brier"] = 0.0
            summary["models"][f"seed_{seed}"]["parse_coverage"] = 1.0

        roster.validate_raw_rows(rows, test_rows, summary, "qwen3_8b")

        summary["models"]["base"]["brier"] += 0.01
        with self.assertRaisesRegex(AssertionError, "raw Brier score drifted"):
            roster.validate_raw_rows(rows, test_rows, summary, "qwen3_8b")


if __name__ == "__main__":
    unittest.main()
