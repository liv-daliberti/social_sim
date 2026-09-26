#!/usr/bin/env python3
"""Select a common matched/prior kernel readout using development episodes only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ANCHORS,
    PRIMARY_ANCHOR,
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    atomic_json,
    file_sha256,
    load_frozen_tasks,
)

ALPHA_MULTIPLIERS = np.asarray((1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0))
ELIGIBLE_LAYERS = tuple(range(1, 36))


def load_states(seed: int, arm: str) -> np.ndarray:
    run_dir = RUNS_DIR / f"qwen3_8b_s{seed}" / arm
    manifest = json.loads((run_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "endpoint_extraction_complete"
        or manifest.get("seed") != seed
        or manifest.get("requested_arms") != [arm]
    ):
        raise ValueError(f"incomplete extraction for seed {seed}")
    path = run_dir / f"states_{arm}.npy"
    return np.load(path, mmap_mode="r", allow_pickle=False)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha_multiplier: float) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    scale = float(np.mean(np.sum(xc * xc, axis=1)) / x.shape[1])
    alpha = float(alpha_multiplier * max(scale, 1e-12))
    kernel = (xc @ xc.T) / x.shape[1]
    dual = np.linalg.solve(kernel + alpha * np.eye(len(x)), yc)
    beta = (xc.T @ dual) / x.shape[1]
    return {"x_mean": x_mean, "y_mean": y_mean, "beta": beta, "alpha": alpha}


def ridge_predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - model["x_mean"]) @ model["beta"] + model["y_mean"]


def arm_mean_features(
    state_files: dict[tuple[int, str], np.ndarray], indices: np.ndarray, layer: int, anchor: int
) -> dict[str, np.ndarray]:
    return {
        arm: np.mean(
            [
                np.asarray(state_files[(seed, arm)][indices, layer, anchor, :], dtype=np.float32)
                for seed in TRAINING_SEEDS
            ],
            axis=0,
        )
        for arm in ("matched", "prior")
    }


def pooled_rows(features: dict[str, np.ndarray], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate((features["matched"], features["prior"]), axis=0),
        np.concatenate((target, target), axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "common_probe")
    args = parser.parse_args()
    tasks, manifest = load_frozen_tasks()
    # The selection surface is frozen to k=0 and contains both original and transplant rows.
    dev_indices = np.asarray(
        [index for index, row in enumerate(tasks) if row["split"] == "dev" and row["k"] == 0]
    )
    if len(dev_indices) != 192 or any(tasks[index]["dev_fold"] is None for index in dev_indices):
        raise ValueError("unexpected development selection surface")
    target_raw = np.asarray([tasks[index]["kernel_h1_h3"] for index in dev_indices], dtype=float)
    target_mean = target_raw.mean(axis=0)
    target_sd = target_raw.std(axis=0, ddof=0)
    if np.any(target_sd <= 0):
        raise ValueError("degenerate development target")
    target = (target_raw - target_mean) / target_sd
    folds = np.asarray([tasks[index]["dev_fold"] for index in dev_indices], dtype=int)
    pair_ids = np.asarray([tasks[index]["donor_pair_id"] for index in dev_indices])
    for pair_id in np.unique(pair_ids):
        if len(set(folds[pair_ids == pair_id])) != 1:
            raise ValueError("a reciprocal donor pair crosses development folds")
    state_files = {
        (seed, arm): load_states(seed, arm)
        for seed in TRAINING_SEEDS
        for arm in ("matched", "prior")
    }
    anchor_index = ANCHORS.index(PRIMARY_ANCHOR)
    cv_rows: list[dict[str, Any]] = []
    cache: dict[int, dict[str, np.ndarray]] = {}
    for layer in ELIGIBLE_LAYERS:
        features = arm_mean_features(state_files, dev_indices, layer, anchor_index)
        cache[layer] = features
        for multiplier in ALPHA_MULTIPLIERS:
            fold_errors = []
            for fold in range(4):
                train = folds != fold
                heldout = folds == fold
                x_train, y_train = pooled_rows(
                    {arm: values[train] for arm, values in features.items()}, target[train]
                )
                fitted = ridge_fit(x_train, y_train, float(multiplier))
                x_test, y_test = pooled_rows(
                    {arm: values[heldout] for arm, values in features.items()}, target[heldout]
                )
                prediction = ridge_predict(fitted, x_test)
                fold_errors.append(float(np.mean((prediction - y_test) ** 2)))
            cv_rows.append(
                {
                    "layer": layer,
                    "alpha_multiplier": float(multiplier),
                    "fold_mse": fold_errors,
                    "mean_mse": float(np.mean(fold_errors)),
                }
            )
        print(f"selected surface evaluated through layer {layer}", flush=True)
    winner = min(cv_rows, key=lambda row: (row["mean_mse"], row["layer"], row["alpha_multiplier"]))
    selected_layer = int(winner["layer"])
    selected_multiplier = float(winner["alpha_multiplier"])
    x_all, y_all = pooled_rows(cache[selected_layer], target)
    final = ridge_fit(x_all, y_all, selected_multiplier)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoder_path = args.output_dir / "decoder.npz"
    np.savez(
        decoder_path,
        x_mean=np.asarray(final["x_mean"], dtype=np.float32),
        y_mean=np.asarray(final["y_mean"], dtype=np.float64),
        beta=np.asarray(final["beta"], dtype=np.float64),
        target_mean=target_mean,
        target_sd=target_sd,
    )
    window = [max(1, min(35, selected_layer + offset)) for offset in (-1, 0, 1)]
    if len(set(window)) != 3:
        window = [1, 2, 3] if selected_layer <= 2 else [33, 34, 35]
    selection = {
        "study": STUDY,
        "status": "development_selection_complete_test_unopened",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks_sha256": manifest["tasks_sha256"],
        "selection_rows": len(dev_indices),
        "selection_episodes": len({tasks[index]["episode_id"] for index in dev_indices}),
        "anchor": PRIMARY_ANCHOR,
        "eligible_layers": list(ELIGIBLE_LAYERS),
        "alpha_multipliers": ALPHA_MULTIPLIERS.tolist(),
        "selected_layer": selected_layer,
        "selected_alpha_multiplier": selected_multiplier,
        "selected_alpha": float(final["alpha"]),
        "selected_cv_mse": float(winner["mean_mse"]),
        "patch_state_layer_window": window,
        "decoder_path": str(decoder_path),
        "decoder_sha256": file_sha256(decoder_path),
        "cv_results": cv_rows,
    }
    atomic_json(args.output_dir / "selection.json", selection)
    print(json.dumps({key: value for key, value in selection.items() if key != "cv_results"}, indent=2))


if __name__ == "__main__":
    main()
