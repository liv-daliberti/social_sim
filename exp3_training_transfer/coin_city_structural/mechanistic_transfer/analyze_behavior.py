#!/usr/bin/env python3
"""Score frozen endpoint behavior and the reference-transplant difference-in-differences."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    atomic_json,
    load_frozen_tasks,
    paired_seed_bootstrap,
)
from worlds import response_vector  # noqa: E402


def parse(text: str) -> np.ndarray | None:
    try:
        payload = json.loads(text.strip())
    except (AttributeError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"forecasts"}:
        return None
    values = payload["forecasts"]
    if not isinstance(values, list) or len(values) != 10:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    return np.clip(np.asarray(values, dtype=float), 40.0, 320.0)


def main() -> None:
    tasks, _ = load_frozen_tasks()
    task_by_id = {row["sample_id"]: row for row in tasks}
    task_scores = {}
    predicted_responses = {}
    parse_by_endpoint = {}
    for seed in TRAINING_SEEDS:
        for arm in ("matched", "prior"):
            path = RUNS_DIR / f"qwen3_8b_s{seed}" / f"behavior_{arm}.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            if len(records) != len(tasks):
                raise ValueError(f"behavior task count mismatch: {path}")
            flags = []
            for record in records:
                task = json.loads(record["reference"])
                frozen = task_by_id[task["sample_id"]]
                truth = np.asarray(frozen["truth_response"], dtype=float)
                errors, responses = [], []
                for output in record["output"]:
                    parsed = parse(output)
                    flags.append(parsed is not None)
                    if parsed is None:
                        errors.append(280.0)
                    else:
                        response = response_vector(parsed)
                        responses.append(response)
                        errors.append(float(np.mean(np.abs(response - truth))))
                task_scores[(seed, arm, frozen["sample_id"])] = float(np.mean(errors))
                predicted_responses[(seed, arm, frozen["sample_id"])] = (
                    None if len(responses) != len(record["output"]) else np.mean(responses, axis=0)
                )
            parse_by_endpoint[f"{seed}:{arm}"] = float(np.mean(flags))

    accuracy: dict[int, np.ndarray] = {}
    matched_tracking: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    prior_tracking: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    did_tracking: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for seed in TRAINING_SEEDS:
        episode_accuracy = []
        test_episodes = sorted({row["episode_id"] for row in tasks if row["split"] == "test"})
        for episode_id in test_episodes:
            contrasts = []
            for k in (0, 8):
                sample_id = f"{episode_id}:k{k}:original"
                contrasts.append(
                    task_scores[(seed, "prior", sample_id)]
                    - task_scores[(seed, "matched", sample_id)]
                )
            episode_accuracy.append(float(np.mean(contrasts)))
        accuracy[seed] = np.asarray(episode_accuracy)
        for k in (0, 8):
            arm_values = {"matched": [], "prior": []}
            for episode_id in test_episodes:
                original = task_by_id[f"{episode_id}:k{k}:original"]
                direction = np.asarray(original["donor_response"]) - np.asarray(
                    original["truth_response"]
                )
                denominator = max(float(np.dot(direction, direction)), 1e-12)
                for arm in ("matched", "prior"):
                    original_prediction = predicted_responses[
                        (seed, arm, f"{episode_id}:k{k}:original")
                    ]
                    transplant_prediction = predicted_responses[
                        (seed, arm, f"{episode_id}:k{k}:transplant")
                    ]
                    if original_prediction is None or transplant_prediction is None:
                        value = 0.0
                    else:
                        value = float(
                            np.dot(transplant_prediction - original_prediction, direction)
                            / denominator
                        )
                    arm_values[arm].append(value)
            matched_tracking[k][seed] = np.asarray(arm_values["matched"])
            prior_tracking[k][seed] = np.asarray(arm_values["prior"])
            did_tracking[k][seed] = (
                matched_tracking[k][seed] - prior_tracking[k][seed]
            )

    summary = {
        "study": STUDY,
        "status": "sealed_behavior_analysis_complete",
        "parse_rate_by_endpoint": parse_by_endpoint,
        "minimum_strict_parse_rate": min(parse_by_endpoint.values()),
        "matched_accuracy_advantage": paired_seed_bootstrap(accuracy, seed=202608291),
        "reference_transplant_by_k": {
            str(k): {
                "matched_tracking": paired_seed_bootstrap(
                    matched_tracking[k], seed=202608292 + k
                ),
                "prior_tracking": paired_seed_bootstrap(
                    prior_tracking[k], seed=202608302 + k
                ),
                "matched_minus_prior_difference_in_differences": paired_seed_bootstrap(
                    did_tracking[k], seed=202608312 + k
                ),
            }
            for k in (0, 8)
        },
    }
    atomic_json(RUNS_DIR / "behavior_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
