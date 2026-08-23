#!/usr/bin/env python3
"""Prompt-only empirical regressions for the coin-to-city analysis.

This module intentionally imports no simulation engine.  Every estimate is
reconstructed from numeric rows displayed in a frozen prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


CITY_BLOCK = re.compile(
    r"CITY ([ABC])\n.*?\n(\| Case .*?)(?=\n\nCITY |\n\nA new City C)",
    re.DOTALL,
)
OBSERVATION_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*50\.0\s*\|\s*\+8\s*\|\s*([0-9.]+)\s*\|$",
    re.MULTILINE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def visible_city_rows(prompt: str) -> dict[str, np.ndarray]:
    blocks = {
        match.group(1): np.asarray(
            [float(value) for value in OBSERVATION_ROW.findall(match.group(2))],
            dtype=float,
        )
        for match in CITY_BLOCK.finditer(prompt)
    }
    if not blocks or any(values.size == 0 for values in blocks.values()):
        raise ValueError("could not reconstruct visible city rows from prompt")
    return blocks


def city_c_intercept_only(prompt: str) -> dict[str, float]:
    """Intercept-only City C regression: mean of visible C outcomes."""
    rows = visible_city_rows(prompt)
    if set(rows) != {"C"}:
        raise ValueError("City C-only prompt unexpectedly contains reference cities")
    return {
        "estimate": float(rows["C"].mean()),
        "city_c_cases": int(rows["C"].size),
    }


def abc_empirical_partial_pool(prompt: str) -> dict[str, float]:
    """Empirical random-intercept shrinkage using displayed A/B/C rows only.

    A and B estimate the reference grand mean, within-city variance, and
    between-city variance by method of moments.  The visible C mean is then
    shrunk toward the empirical reference grand mean.  There are no fixed DGP
    variances, latent type labels, contextual inputs, or future C outcomes.
    """
    rows = visible_city_rows(prompt)
    if set(rows) != {"A", "B", "C"}:
        raise ValueError("A/B/C prompt is missing a displayed city")
    a, b, c = rows["A"], rows["B"], rows["C"]
    if a.size < 2 or b.size < 2:
        raise ValueError("reference cities need repeated displayed cases")
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    grand_mean = 0.5 * (mean_a + mean_b)
    within_df = int(a.size + b.size - 2)
    within_ss = float(np.sum((a - mean_a) ** 2) + np.sum((b - mean_b) ** 2))
    within_variance = within_ss / within_df
    reference_mean_sample_variance = (
        (mean_a - grand_mean) ** 2 + (mean_b - grand_mean) ** 2
    )
    mean_sampling_variance = 0.5 * within_variance * (
        1.0 / float(a.size) + 1.0 / float(b.size)
    )
    between_variance = max(
        0.0, reference_mean_sample_variance - mean_sampling_variance
    )
    city_c_mean = float(c.mean())
    city_c_sampling_variance = within_variance / float(c.size)
    denominator = between_variance + city_c_sampling_variance
    city_c_weight = between_variance / denominator if denominator > 0 else 1.0
    estimate = city_c_weight * city_c_mean + (1.0 - city_c_weight) * grand_mean
    return {
        "estimate": float(estimate),
        "city_c_cases": int(c.size),
        "reference_a_cases": int(a.size),
        "reference_b_cases": int(b.size),
        "reference_grand_mean": grand_mean,
        "within_variance_estimate": within_variance,
        "between_variance_estimate": between_variance,
        "city_c_weight": city_c_weight,
    }


def empirical_analysis_keys(
    design_dir: Path,
    original_keys: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return scoring keys whose regression references are prompt-only."""
    baseline_tasks = {
        row["task_id"]: row
        for row in _read_jsonl(design_dir / "tasks_baseline.jsonl")
    }
    abc_tasks = {
        row["task_id"]: row
        for row in _read_jsonl(design_dir / "tasks_abc_no_context.jsonl")
    }
    if set(baseline_tasks) != set(original_keys) or set(abc_tasks) != set(original_keys):
        raise ValueError("task IDs do not match the scoring key")
    analysis_keys: dict[str, dict[str, Any]] = {}
    records = []
    for task_id, original in original_keys.items():
        city_c = city_c_intercept_only(baseline_tasks[task_id]["prompt"])
        abc = abc_empirical_partial_pool(abc_tasks[task_id]["prompt"])
        if city_c["city_c_cases"] != original["city_c_cases"]:
            raise ValueError(f"visible C row count mismatch for {task_id}")
        # Build a deliberately narrow scoring record.  Latent design labels
        # and the simulator's original baselines are never passed downstream.
        revised = {
            "task_id": task_id,
            "episode": original["episode"],
            "k": original["k"],
            "city_c_cases": city_c["city_c_cases"],
            "gold_expected_poll": original["gold_expected_poll"],
            "baselines": {
                "baseline": city_c["estimate"],
                "abc_no_context": abc["estimate"],
            # No context-aware regression is displayed.  This alias ensures
            # even hidden plotting ranges contain no privileged benchmark.
                "abc_context": abc["estimate"],
            },
        }
        analysis_keys[task_id] = revised
        records.append(
            {
                "task_id": task_id,
                "episode": original["episode"],
                "k": original["k"],
                "gold_expected_poll_used_for_scoring_only": original[
                    "gold_expected_poll"
                ],
                "city_c_regression": city_c,
                "abc_no_context_empirical_partial_pool": abc,
            }
        )
    return analysis_keys, records
