#!/usr/bin/env python3
"""Fit leakage-safe layer-wise probes to Experiment 3 endpoint states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
sys.path.insert(0, str(FAMILY))
from worlds import BY_NAME  # noqa: E402

DEFAULT_TASK_DIR = HERE / "data"
DEFAULT_RUN_DIR = HERE / "runs" / "qwen3_8b_base"
DISCLOSURES = ("disclosed", "undisclosed")
K_VALUES = (3, 6, 9)
TARGETS = ("gain", "response_residual")
ALPHAS = np.asarray((1e-4, 3e-3, 1e-1, 3.0, 100.0), dtype=float)
PERMUTATION_SEED = 20260824


@dataclass(frozen=True)
class DualDesign:
    train_kernel: np.ndarray
    eval_kernel: np.ndarray


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


def prepare_dual(train: np.ndarray, evaluate: np.ndarray) -> DualDesign:
    train = np.asarray(train, dtype=np.float64)
    evaluate = np.asarray(evaluate, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalizer = math.sqrt(train.shape[1])
    train_z = ((train - mean) / scale) / normalizer
    eval_z = ((evaluate - mean) / scale) / normalizer
    return DualDesign(train_z @ train_z.T, eval_z @ train_z.T)


def dual_predict(design: DualDesign, target: np.ndarray, alpha: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    center = target.mean(axis=0)
    system = design.train_kernel + float(alpha) * np.eye(len(target))
    coefficients = np.linalg.solve(system, target - center)
    return design.eval_kernel @ coefficients + center


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
        prediction = prediction[:, None]
    denominator = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    worlds: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        error = target - prediction
    else:
        error = target - prediction
    by_world = {}
    for world in sorted(set(worlds.tolist())):
        mask = worlds == world
        by_world[world] = {
            "n": int(mask.sum()),
            "r2": r2_score(prediction[mask], target[mask]),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
            "mae": float(np.mean(np.abs(error[mask]))),
        }
    finite = [item["r2"] for item in by_world.values() if np.isfinite(item["r2"])]
    return {
        "n": len(target),
        "macro_within_world_r2": float(np.mean(finite)) if finite else float("nan"),
        "overall_r2": r2_score(prediction, target),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "by_world": by_world,
    }


def target_values(rows: list[dict[str, Any]], target_name: str) -> np.ndarray:
    if target_name == "gain":
        values = []
        for row in rows:
            world = BY_NAME[row["world"]]
            midpoint = (world.gain_lo + world.gain_hi) / 2.0
            width = world.gain_hi - world.gain_lo
            values.append((float(row["target_gain"]) - midpoint) / width)
        return np.asarray(values, dtype=float)
    if target_name == "response_residual":
        return np.asarray(
            [
                (np.asarray(row["truth_response"]) - np.asarray(row["prior_response"]))
                / 4.0
                for row in rows
            ],
            dtype=float,
        )
    raise ValueError(target_name)


def score_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
) -> float:
    worlds = np.asarray([row["world"] for row in rows])
    return regression_metrics(prediction, target, worlds)["macro_within_world_r2"]


def cross_validated_alpha(
    *,
    train: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
    folds: np.ndarray,
    alphas: np.ndarray = ALPHAS,
) -> tuple[float, float, dict[str, float]]:
    unique_folds = sorted(set(int(value) for value in folds))
    designs = []
    for fold in unique_folds:
        validation = folds == fold
        designs.append((validation, prepare_dual(train[~validation], train[validation])))
    scores = {}
    for alpha in alphas:
        prediction = np.empty_like(target, dtype=float)
        for fold, (validation, design) in zip(unique_folds, designs):
            fitting = folds != fold
            prediction[validation] = dual_predict(design, target[fitting], float(alpha))
        scores[f"{alpha:g}"] = score_predictions(prediction, target, rows)
    best_alpha = max(
        (float(alpha) for alpha in alphas),
        key=lambda alpha: (
            scores[f"{alpha:g}"] if np.isfinite(scores[f"{alpha:g}"]) else -np.inf,
            -alpha,
        ),
    )
    return best_alpha, scores[f"{best_alpha:g}"], scores


def permute_within_world(target: np.ndarray, rows: list[dict[str, Any]], rng) -> np.ndarray:
    permuted = np.array(target, copy=True)
    worlds = np.asarray([row["world"] for row in rows])
    for world in sorted(set(worlds.tolist())):
        indices = np.flatnonzero(worlds == world)
        permuted[indices] = target[rng.permutation(indices)]
    return permuted


def permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    observed = score_predictions(prediction, target, rows)
    rng = np.random.default_rng(PERMUTATION_SEED)
    scores = np.asarray(
        [
            score_predictions(prediction, permute_within_world(target, rows, rng), rows)
            for _ in range(repeats)
        ],
        dtype=float,
    )
    finite = scores[np.isfinite(scores)]
    p_value = (
        float("nan")
        if not len(finite) or not np.isfinite(observed)
        else float((1 + np.sum(finite >= observed)) / (len(finite) + 1))
    )
    return {
        "scheme": (
            "held-out row permutation within world; development-selected layer, "
            "ridge strength, and fitted probe held fixed"
        ),
        "repeats": repeats,
        "seed": PERMUTATION_SEED,
        "observed_score": observed,
        "null_mean": float(finite.mean()) if len(finite) else float("nan"),
        "null_q025": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "null_q975": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "one_sided_p": p_value,
    }


def indices_for(
    tasks: list[dict[str, Any]],
    *,
    disclosure: str,
    k: int,
    split: str,
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(tasks)
            if row["disclosure"] == disclosure and row["k"] == k and row["split"] == split
        ],
        dtype=int,
    )


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_representation(
    *,
    name: str,
    features: np.ndarray,
    tasks: list[dict[str, Any]],
    permutation_repeats: int,
    layerwise: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if features.ndim == 2:
        features = features[:, None, :]
    layer_count = features.shape[1]
    result: dict[str, Any] = {
        "representation": name,
        "layer_count": layer_count,
        "conditions": {},
    }
    csv_rows = []

    for disclosure in DISCLOSURES:
        opposite = "undisclosed" if disclosure == "disclosed" else "disclosed"
        for k in K_VALUES:
            dev_indices = indices_for(tasks, disclosure=disclosure, k=k, split="dev")
            test_indices = indices_for(tasks, disclosure=disclosure, k=k, split="test")
            cross_indices = indices_for(tasks, disclosure=opposite, k=k, split="test")
            dev_rows = [tasks[index] for index in dev_indices]
            test_rows = [tasks[index] for index in test_indices]
            cross_rows = [tasks[index] for index in cross_indices]
            folds = np.asarray([row["dev_fold"] for row in dev_rows], dtype=int)
            for target_name in TARGETS:
                dev_target = target_values(dev_rows, target_name)
                test_target = target_values(test_rows, target_name)
                cross_target = target_values(cross_rows, target_name)
                layers = []
                for layer_index in range(layer_count):
                    alpha, cv_score, alpha_scores = cross_validated_alpha(
                        train=features[dev_indices, layer_index, :],
                        target=dev_target,
                        rows=dev_rows,
                        folds=folds,
                    )
                    item = {
                        "layer": layer_index,
                        "alpha": alpha,
                        "cv_score": cv_score,
                        "alpha_scores": alpha_scores,
                    }
                    layers.append(item)
                    csv_rows.append(
                        {
                            "representation": name,
                            "disclosure": disclosure,
                            "k": k,
                            "target": target_name,
                            "layer": layer_index,
                            "alpha": alpha,
                            "cv_macro_within_world_r2": cv_score,
                            "selected": False,
                        }
                    )
                selected_layer = max(
                    range(layer_count),
                    key=lambda layer: (
                        layers[layer]["cv_score"]
                        if np.isfinite(layers[layer]["cv_score"])
                        else -np.inf,
                        -layer,
                    ),
                )
                selected = layers[selected_layer]
                train_features = features[dev_indices, selected_layer, :]
                combined_eval = np.concatenate(
                    (
                        features[test_indices, selected_layer, :],
                        features[cross_indices, selected_layer, :],
                    ),
                    axis=0,
                )
                design = prepare_dual(train_features, combined_eval)
                prediction = dual_predict(design, dev_target, selected["alpha"])
                same_prediction = prediction[: len(test_indices)]
                cross_prediction = prediction[len(test_indices) :]
                test_worlds = np.asarray([row["world"] for row in test_rows])
                cross_worlds = np.asarray([row["world"] for row in cross_rows])
                key = f"{disclosure}:k{k}:{target_name}"
                result["conditions"][key] = {
                    "train_disclosure": disclosure,
                    "cross_disclosure": opposite,
                    "k": k,
                    "target": target_name,
                    "development_n": len(dev_rows),
                    "test_n": len(test_rows),
                    "selected_layer": selected_layer,
                    "selected_alpha": selected["alpha"],
                    "selected_cv_macro_within_world_r2": selected["cv_score"],
                    "layer_selection_source": "development CV only",
                    "same_disclosure_test": regression_metrics(
                        same_prediction, test_target, test_worlds
                    ),
                    "cross_disclosure_test": regression_metrics(
                        cross_prediction, cross_target, cross_worlds
                    ),
                    "permutation": permutation_null(
                        same_prediction,
                        test_target,
                        test_rows,
                        permutation_repeats,
                    ),
                    "layers": layers if layerwise else None,
                }
                for row in reversed(csv_rows):
                    if (
                        row["representation"] == name
                        and row["disclosure"] == disclosure
                        and row["k"] == k
                        and row["target"] == target_name
                        and row["layer"] == selected_layer
                    ):
                        row["selected"] = True
                        break
    return result, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--status", default="probe_complete")
    args = parser.parse_args()

    tasks = read_jsonl(args.task_dir / "tasks.jsonl")
    task_manifest = json.loads(
        (args.task_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    if file_sha256(args.task_dir / "tasks.jsonl") != task_manifest["tasks_sha256"]:
        raise ValueError("tasks hash mismatch")
    extraction_manifest = json.loads(
        (args.run_dir / "extraction_manifest.json").read_text(encoding="utf-8")
    )
    features_path = args.run_dir / "hidden_states.npz"
    if file_sha256(features_path) != extraction_manifest["features_sha256"]:
        raise ValueError("hidden-state hash mismatch")
    with np.load(features_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        last_token = payload["last_token"].astype(np.float32)
        mean_control = payload["mean_control"].astype(np.float32)
    if sample_ids != [row["sample_id"] for row in tasks]:
        raise ValueError("feature/task order mismatch")

    representations = {}
    csv_rows = []
    primary, primary_rows = analyze_representation(
        name="final_input_token_layerwise",
        features=last_token,
        tasks=tasks,
        permutation_repeats=args.permutations,
        layerwise=True,
    )
    representations[primary["representation"]] = primary
    csv_rows.extend(primary_rows)
    for control_index, name in enumerate(("mean_embedding", "mean_final_layer")):
        control, control_rows = analyze_representation(
            name=name,
            features=mean_control[:, control_index, :],
            tasks=tasks,
            permutation_repeats=args.permutations,
            layerwise=False,
        )
        representations[name] = control
        csv_rows.extend(control_rows)

    result = clean_json(
        {
            "study": task_manifest["study"],
            "status": args.status,
            "model": extraction_manifest["model"],
            "model_label": extraction_manifest["model_label"],
            "endpoint_label": extraction_manifest["endpoint_label"],
            "task_manifest_sha256": file_sha256(
                args.task_dir / "task_manifest.json"
            ),
            "extraction_manifest_sha256": file_sha256(
                args.run_dir / "extraction_manifest.json"
            ),
            "features_sha256": extraction_manifest["features_sha256"],
            "targets": {
                "gain": "generator-range-scaled episode-specific gain",
                "response_residual": (
                    "(truth impulse response - registered population-prior response) / 4"
                ),
            },
            "primary_score": "macro mean of held-out within-world variance-weighted R2",
            "representations": representations,
        }
    )
    result_path = args.run_dir / "probe_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = args.run_dir / "layerwise_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                "probe_results": str(result_path),
                "layerwise_results": str(csv_path),
                "conditions": len(primary["conditions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
