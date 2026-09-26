#!/usr/bin/env python3
"""Analyze six models on the same frozen Coin City tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
DEFAULT_OUTPUT = RUN / "analysis" / "multimodel_results.json"

MODELS = (
    "claude-opus-4-8",
    "gpt-5.6-sol",
    "DeepSeek-V4-Pro",
    "FW-Kimi-K3",
    "gemini-3.6-flash",
    "claude-opus-5",
)
ARMS = ("baseline", "abc_no_context", "abc_context")
EXPECTED_TASKS_PER_ARM = 1_250


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_responses(model: str, arm: str) -> dict[str, dict]:
    path = RESPONSES / f"responses_{model}_{arm}.jsonl"
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        old = latest.get(row["task_id"])
        if old is None or row.get("predicted_poll") is not None:
            latest[row["task_id"]] = row
    return latest


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci_95": [float(low), float(high)],
    }


def analyze_model(
    model: str,
    *,
    scores: dict[str, dict],
    episodes: dict[int, dict],
    bootstrap_draws: int,
) -> dict:
    responses = {arm: latest_responses(model, arm) for arm in ARMS}
    record_counts = {
        arm: len(read_jsonl(RESPONSES / f"responses_{model}_{arm}.jsonl"))
        if (RESPONSES / f"responses_{model}_{arm}.jsonl").exists()
        else 0
        for arm in ARMS
    }
    parsed_counts = {
        arm: sum(row.get("predicted_poll") is not None for row in rows.values())
        for arm, rows in responses.items()
    }
    curves: dict[str, dict] = {}
    for k in range(5):
        task_ids = {
            task_id
            for task_id, score in scores.items()
            if int(score["c_cases"]) == k
        }
        common = sorted(
            task_id
            for task_id in task_ids
            if all(
                responses[arm].get(task_id, {}).get("predicted_poll") is not None
                for arm in ARMS
            )
        )
        if not common:
            curves[str(k)] = {"n": 0}
            continue
        truths = np.asarray(
            [scores[task_id]["gold_expected_poll"] for task_id in common],
            dtype=float,
        )
        arm_errors: dict[str, np.ndarray] = {}
        arm_metrics: dict[str, dict] = {}
        for arm in ARMS:
            predictions = np.asarray(
                [responses[arm][task_id]["predicted_poll"] for task_id in common],
                dtype=float,
            )
            arm_errors[arm] = np.abs(predictions - truths)
            implied = np.asarray(
                [
                    (
                        prediction
                        - episodes[scores[task_id]["episode"]]["query_starting_poll"]
                    )
                    / episodes[scores[task_id]["episode"]]["query_net_news"]
                    for task_id, prediction in zip(common, predictions)
                ],
                dtype=float,
            )
            target_slopes = np.asarray(
                [
                    episodes[scores[task_id]["episode"]]["target_slope"]
                    for task_id in common
                ],
                dtype=float,
            )
            arm_metrics[arm] = {
                "forecast_mae": float(np.mean(arm_errors[arm])),
                "slope_mae": float(np.mean(np.abs(implied - target_slopes))),
                "rho": correlation(implied, target_slopes),
            }

        abc_predictions = np.asarray(
            [scores[task_id]["baselines"]["abc_no_context"] for task_id in common],
            dtype=float,
        )
        abc_errors = np.abs(abc_predictions - truths)
        seed_prefix = int(hashlib.sha256(model.encode()).hexdigest()[:8], 16)
        curves[str(k)] = {
            "n": len(common),
            "arms": arm_metrics,
            "abc_ols_forecast_mae": float(np.mean(abc_errors)),
            "paired_forecast_mae": {
                "context_minus_no_context": bootstrap_mean_ci(
                    arm_errors["abc_context"] - arm_errors["abc_no_context"],
                    seed=seed_prefix + 10 * k + 1,
                    draws=bootstrap_draws,
                ),
                "context_minus_target_only": bootstrap_mean_ci(
                    arm_errors["abc_context"] - arm_errors["baseline"],
                    seed=seed_prefix + 10 * k + 2,
                    draws=bootstrap_draws,
                ),
                "context_minus_abc_ols": bootstrap_mean_ci(
                    arm_errors["abc_context"] - abc_errors,
                    seed=seed_prefix + 10 * k + 3,
                    draws=bootstrap_draws,
                ),
            },
        }
    return {
        "record_counts": record_counts,
        "parsed_counts": parsed_counts,
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--stdout-only", action="store_true")
    args = parser.parse_args()

    scores = {
        row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")
    }
    episodes = {
        row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")
    }
    result = {
        "experiment": RUN.name,
        "models": {
            model: analyze_model(
                model,
                scores=scores,
                episodes=episodes,
                bootstrap_draws=args.bootstrap_draws,
            )
            for model in MODELS
        },
        "bootstrap_draws": args.bootstrap_draws,
    }
    complete = all(
        model_data["record_counts"][arm] == EXPECTED_TASKS_PER_ARM
        for model_data in result["models"].values()
        for arm in ARMS
    )
    result["complete"] = complete
    if not complete and not args.allow_partial:
        raise SystemExit("multi-model response set is incomplete")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
