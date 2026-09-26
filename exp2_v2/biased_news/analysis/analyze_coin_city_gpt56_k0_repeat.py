#!/usr/bin/env python3
"""Paired run-to-run audit for the registered GPT-5.6 Coin City repeat."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_coin_city_symbol_context import (
    bootstrap_mean_ci,
    bootstrap_metric_difference,
    correlation,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
DESIGN = RUN / "design"
REPEAT = RUN / "responses" / "gpt56_k0_repeat_20260826"
COMPARISON = (
    ROOT
    / "local_results"
    / "symbol_context_model_comparison_20260824"
    / "exploratory_results.json"
)
OUTPUT = RUN / "analysis" / "gpt56_k0_repeat_v1.json"
MODEL = "gpt-5.6-sol"
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
BOOTSTRAP_DRAWS = 5_000


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        previous = rows.get(row["task_id"])
        if previous is None or row.get("predicted_poll") is not None:
            rows[row["task_id"]] = row
    return rows


def implied(
    predictions: np.ndarray, task_ids: list[str], scores: dict, episodes: dict
) -> np.ndarray:
    return np.asarray(
        [
            (prediction - episodes[scores[task_id]["episode"]]["query_starting_poll"])
            / episodes[scores[task_id]["episode"]]["query_net_news"]
            for task_id, prediction in zip(task_ids, predictions)
        ],
        dtype=float,
    )


def metric_record(
    predictions: np.ndarray,
    slopes: np.ndarray,
    truths: np.ndarray,
    target_slopes: np.ndarray,
) -> dict:
    return {
        "forecast_mae": float(np.mean(np.abs(predictions - truths))),
        "slope_mae": float(np.mean(np.abs(slopes - target_slopes))),
        "rho": correlation(slopes, target_slopes),
        "regime_accuracy": float(
            np.mean((slopes >= 0.575) == (target_slopes >= 0.575))
        ),
    }


def bootstrap_contrast_of_contrasts(
    repeat_left: np.ndarray,
    repeat_right: np.ndarray,
    original_left: np.ndarray,
    original_right: np.ndarray,
    target: np.ndarray,
    *,
    metric: str,
    seed: int,
) -> dict:
    def evaluate(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> float:
        if metric == "correlation":
            left_value = correlation(left, y)
            right_value = correlation(right, y)
            if left_value is None or right_value is None:
                return float("nan")
            return left_value - right_value
        if metric == "regime_accuracy":
            return float(np.mean((left >= 0.575) == (y >= 0.575))) - float(
                np.mean((right >= 0.575) == (y >= 0.575))
            )
        raise ValueError(metric)

    point = evaluate(repeat_left, repeat_right, target) - evaluate(
        original_left, original_right, target
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(target), size=len(target))
        value = evaluate(
            repeat_left[indices], repeat_right[indices], target[indices]
        ) - evaluate(original_left[indices], original_right[indices], target[indices])
        if np.isfinite(value):
            values.append(value)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "difference_in_differences": float(point),
        "ci_95": [float(low), float(high)],
    }


def main() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    scores = {row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")}
    episodes = {
        int(row["episode"]): row for row in read_jsonl(DESIGN / "episodes.jsonl")
    }
    original_paths = {
        arm: Path(comparison["response_paths"][f"{MODEL}:{arm}"]) for arm in ARMS
    }
    repeat_paths = {arm: REPEAT / f"responses_{MODEL}_{arm}.jsonl" for arm in ARMS}
    if not all(path.exists() for path in repeat_paths.values()):
        raise SystemExit("GPT-5.6 repeat response set is incomplete")

    original = {arm: latest(path) for arm, path in original_paths.items()}
    repeat = {arm: latest(path) for arm, path in repeat_paths.items()}
    validation = {}
    for arm in ARMS:
        rows = read_jsonl(repeat_paths[arm])
        unique = {row["task_id"]: row for row in rows}
        expected = {
            task_id for task_id, score in scores.items() if int(score["c_cases"]) == 0
        }
        invalid = [
            task_id
            for task_id, row in unique.items()
            if row.get("arm") != arm
            or row.get("model") != MODEL
            or row.get("attempts") != 1
            or task_id not in expected
        ]
        if len(rows) != 250 or set(unique) != expected or invalid:
            raise SystemExit(f"invalid repeat shard for {arm}")
        validation[arm] = {
            "records": len(rows),
            "unique_tasks": len(unique),
            "parsed": sum(
                row.get("predicted_poll") is not None for row in unique.values()
            ),
            "transport_failures": sum(
                not row.get("response_received", False) for row in unique.values()
            ),
            "unparseable": sum(
                row.get("response_received", False)
                and row.get("predicted_poll") is None
                for row in unique.values()
            ),
        }

    common = sorted(
        task_id
        for task_id, score in scores.items()
        if int(score["c_cases"]) == 0
        and all(
            collection[arm].get(task_id, {}).get("predicted_poll") is not None
            for collection in (original, repeat)
            for arm in ARMS
        )
    )
    if not common:
        raise SystemExit("no jointly parsed repeat tasks")
    truths = np.asarray(
        [scores[task_id]["gold_expected_poll"] for task_id in common], dtype=float
    )
    target_slopes = np.asarray(
        [episodes[scores[task_id]["episode"]]["target_slope"] for task_id in common],
        dtype=float,
    )
    predictions = {
        run: {
            arm: np.asarray(
                [rows[arm][task_id]["predicted_poll"] for task_id in common],
                dtype=float,
            )
            for arm in ARMS
        }
        for run, rows in (("original", original), ("repeat", repeat))
    }
    slopes = {
        run: {
            arm: implied(values, common, scores, episodes)
            for arm, values in arms.items()
        }
        for run, arms in predictions.items()
    }

    per_arm = {}
    for index, arm in enumerate(ARMS):
        original_error = np.abs(predictions["original"][arm] - truths)
        repeat_error = np.abs(predictions["repeat"][arm] - truths)
        absolute_delta = np.abs(
            predictions["repeat"][arm] - predictions["original"][arm]
        )
        per_arm[arm] = {
            "n_joint": len(common),
            "original": metric_record(
                predictions["original"][arm],
                slopes["original"][arm],
                truths,
                target_slopes,
            ),
            "repeat": metric_record(
                predictions["repeat"][arm], slopes["repeat"][arm], truths, target_slopes
            ),
            "prediction_absolute_delta": {
                "mean": float(np.mean(absolute_delta)),
                "median": float(np.median(absolute_delta)),
                "p90": float(np.quantile(absolute_delta, 0.9)),
                "exact_match_rate": float(np.mean(absolute_delta == 0)),
            },
            "repeat_minus_original_forecast_mae": bootstrap_mean_ci(
                repeat_error - original_error,
                seed=20260826 + index,
                draws=BOOTSTRAP_DRAWS,
            ),
            "repeat_minus_original_rho": bootstrap_metric_difference(
                slopes["repeat"][arm],
                slopes["original"][arm],
                target_slopes,
                metric="correlation",
                seed=20260836 + index,
                draws=BOOTSTRAP_DRAWS,
            ),
            "repeat_minus_original_regime_accuracy": bootstrap_metric_difference(
                slopes["repeat"][arm],
                slopes["original"][arm],
                target_slopes,
                metric="regime_accuracy",
                seed=20260846 + index,
                draws=BOOTSTRAP_DRAWS,
            ),
        }

    contrasts = {}
    for left in ("abc_context", "abc_symbol_context"):
        right = "abc_no_context"
        repeat_error_effect = np.abs(predictions["repeat"][left] - truths) - np.abs(
            predictions["repeat"][right] - truths
        )
        original_error_effect = np.abs(predictions["original"][left] - truths) - np.abs(
            predictions["original"][right] - truths
        )
        tag = f"{left}_minus_{right}"
        seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
        contrasts[tag] = {
            "original_forecast_mae": bootstrap_mean_ci(
                original_error_effect, seed=seed, draws=BOOTSTRAP_DRAWS
            ),
            "repeat_forecast_mae": bootstrap_mean_ci(
                repeat_error_effect, seed=seed + 1, draws=BOOTSTRAP_DRAWS
            ),
            "repeat_minus_original_forecast_mae": bootstrap_mean_ci(
                repeat_error_effect - original_error_effect,
                seed=seed + 2,
                draws=BOOTSTRAP_DRAWS,
            ),
            "repeat_minus_original_rho_effect": bootstrap_contrast_of_contrasts(
                slopes["repeat"][left],
                slopes["repeat"][right],
                slopes["original"][left],
                slopes["original"][right],
                target_slopes,
                metric="correlation",
                seed=seed + 3,
            ),
            "repeat_minus_original_regime_accuracy_effect": bootstrap_contrast_of_contrasts(
                slopes["repeat"][left],
                slopes["repeat"][right],
                slopes["original"][left],
                slopes["original"][right],
                target_slopes,
                metric="regime_accuracy",
                seed=seed + 4,
            ),
        }

    report = {
        "protocol_version": "coin_city_robustness_v1",
        "classification": "post_hoc_run_to_run_repeat",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "deployment_settings": {
            "reasoning_effort": "low",
            "provider_native_sampling": True,
            "temperature_argument": 0,
            "application_attempts": 1,
            "http_client_retries": 0,
        },
        "bootstrap_unit": "episode",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "jointly_parsed_episodes": len(common),
        "validation": validation,
        "per_arm": per_arm,
        "paired_arm_effects": contrasts,
        "sources": {
            "original": {
                arm: str(path.resolve()) for arm, path in original_paths.items()
            },
            "repeat": {arm: str(path.resolve()) for arm, path in repeat_paths.items()},
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "jointly_parsed": len(common),
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
