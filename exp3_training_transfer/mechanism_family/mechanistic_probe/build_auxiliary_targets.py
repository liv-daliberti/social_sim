#!/usr/bin/env python3
"""Build prompt-derived numerical controls for the Exp3 hidden-state probe.

The frozen prompt/task file is not changed.  This sidecar is keyed by sample ID
and can therefore be added after extraction without changing a hidden-state
artifact or its ordering.  All oracle quantities use only information displayed
in the corresponding prompt condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
DEFAULT_TASK_DIR = HERE / "data"
DEFAULT_OUTPUT = DEFAULT_TASK_DIR / "auxiliary_targets_v2.jsonl"
sys.path.insert(0, str(FAMILY))
from worlds import (  # noqa: E402
    BY_NAME,
    CATALOG,
    GAIN_GRID,
    forecast_scenarios,
    log_likelihood,
    response_vector,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        return 0.0
    return float(centered @ (y - y.mean()) / denominator)


def normalized_gain(world, gain: float) -> float:
    midpoint = (world.gain_lo + world.gain_hi) / 2.0
    return (float(gain) - midpoint) / (world.gain_hi - world.gain_lo)


def known_world_oracle(reference: dict[str, Any]) -> tuple[float, np.ndarray]:
    """Posterior mean using the disclosed mechanism and disclosed gain range."""
    world = BY_NAME[reference["world"]]
    grid = GAIN_GRID[(GAIN_GRID >= world.gain_lo) & (GAIN_GRID <= world.gain_hi)]
    inputs = np.asarray(reference["target_inputs"], dtype=float)
    observed = np.asarray(reference["target_observed"], dtype=float)
    ll = np.asarray(
        [log_likelihood(world, inputs, observed, float(gain)) for gain in grid]
    )
    weights = np.exp(ll - ll.max())
    weights /= weights.sum()
    forecasts = np.stack(
        [forecast_scenarios(world, inputs, float(gain)) for gain in grid]
    )
    return float(weights @ grid), weights @ forecasts


def blind_oracle(reference: dict[str, Any]) -> tuple[float, np.ndarray]:
    """Joint structure/gain posterior using only undisclosed-prompt evidence."""
    surface = BY_NAME[reference["world"]]
    calibrations = reference["calibrations"]
    target_inputs = np.asarray(reference["target_inputs"], dtype=float)
    target_observed = np.asarray(reference["target_observed"], dtype=float)
    weighted: list[tuple[float, float, np.ndarray]] = []
    for catalog_world in CATALOG:
        if catalog_world.drivers != surface.drivers:
            continue
        candidate = catalog_world.on_surface_of(surface)
        structure_ll = 0.0
        for calibration in calibrations:
            values = np.asarray(
                [
                    log_likelihood(
                        candidate,
                        np.asarray(calibration["inputs"], dtype=float),
                        np.asarray(calibration["observed"], dtype=float),
                        float(gain),
                    )
                    for gain in GAIN_GRID
                ]
            )
            peak = float(values.max())
            structure_ll += peak + float(np.log(np.mean(np.exp(values - peak))))
        for gain in GAIN_GRID:
            ll = structure_ll + log_likelihood(
                candidate, target_inputs, target_observed, float(gain)
            )
            weighted.append(
                (
                    ll,
                    float(gain),
                    forecast_scenarios(candidate, target_inputs, float(gain)),
                )
            )
    log_weights = np.asarray([item[0] for item in weighted])
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    gain = float(weights @ np.asarray([item[1] for item in weighted]))
    forecast = weights @ np.stack([item[2] for item in weighted])
    return gain, forecast


def load_references() -> dict[tuple[str, str], dict[str, Any]]:
    from datasets import load_from_disk

    references = {}
    for disclosure in ("disclosed", "undisclosed"):
        path = FAMILY / "data" / f"{disclosure}_causal_family" / "heldout"
        for item in load_from_disk(str(path))["train"]:
            reference = json.loads(item["reference"])
            references[(disclosure, reference["task_id"])] = reference
    return references


def make_row(task: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    world = BY_NAME[reference["world"]]
    inputs = np.asarray(reference["target_inputs"], dtype=float)
    observed = np.asarray(reference["target_observed"], dtype=float)
    oracle_gain, oracle_forecast = (
        known_world_oracle(reference)
        if task["disclosure"] == "disclosed"
        else blind_oracle(reference)
    )
    primary_driver = inputs[:, 0]
    prior_response = np.asarray(reference["prior_response"], dtype=float)
    oracle_response = response_vector(oracle_forecast)
    return {
        "sample_id": task["sample_id"],
        "task_id": task["task_id"],
        "episode_id": task["episode_id"],
        "world": task["world"],
        "k": int(task["k"]),
        "disclosure": task["disclosure"],
        "split": task["split"],
        "last_target_observation": float(observed[-1]),
        "target_observation_mean": float(observed.mean()),
        "target_time_slope": slope(np.arange(len(observed), dtype=float), observed),
        "displayed_driver_outcome_slope": slope(primary_driver, observed),
        "oracle_posterior_gain": normalized_gain(world, oracle_gain),
        "oracle_response_residual": ((oracle_response - prior_response) / 4.0).tolist(),
    }


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks_path = args.task_dir / "tasks.jsonl"
    tasks = read_jsonl(tasks_path)
    references = load_references()
    rows = []
    for task in tasks:
        key = (task["disclosure"], task["task_id"])
        if key not in references:
            raise KeyError(f"missing held-out reference for {key}")
        rows.append(clean(make_row(task, references[key])))
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in tasks]:
        raise AssertionError("auxiliary target order drifted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "auxiliary_targets_complete",
                "rows": len(rows),
                "tasks_sha256": file_sha256(tasks_path),
                "output": str(args.output),
                "output_sha256": file_sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
