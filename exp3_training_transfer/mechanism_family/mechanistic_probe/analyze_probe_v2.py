#!/usr/bin/env python3
"""Leakage-safe v2 analysis for the Experiment 3 latent-mechanism probe.

Version 2 preserves the frozen prompts and v1 results while adding:

* fold-local development centering by world (and by world x k when pooled),
* a fair development-estimated baseline alongside oracle-test-mean R2,
* pooled evidence-depth probes with episode-grouped folds,
* prompt-derived numerical positive controls from a frozen sidecar,
* mean-aligned cross-disclosure transfer using unlabeled development pairs, and
* row-level sealed predictions for paired bootstrap analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TASK_DIR = HERE / "data"
DEFAULT_RUN_DIR = HERE / "runs" / "qwen3_8b_base"
DEFAULT_AUXILIARY = DEFAULT_TASK_DIR / "auxiliary_targets_v2.jsonl"
DISCLOSURES = ("disclosed", "undisclosed")
K_VALUES = (3, 6, 9)
PRIMARY_TARGETS = ("gain", "response_residual")
CONTROL_TARGETS = (
    "last_target_observation",
    "target_observation_mean",
    "target_time_slope",
    "displayed_driver_outcome_slope",
    "oracle_posterior_gain",
)
ALPHAS = np.asarray((1e-4, 3e-3, 1e-1, 3.0, 100.0), dtype=float)
PERMUTATION_SEED = 20260824


def load_v1():
    spec = importlib.util.spec_from_file_location("exp3_probe_v1", HERE / "analyze_probe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v1 probe helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_v1()


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


def attach_auxiliary(
    tasks: list[dict[str, Any]], auxiliary_path: Path
) -> list[dict[str, Any]]:
    auxiliary = read_jsonl(auxiliary_path)
    if [row["sample_id"] for row in auxiliary] != [row["sample_id"] for row in tasks]:
        raise ValueError("auxiliary-target/task order mismatch")
    return [
        {**task, **{name: control[name] for name in CONTROL_TARGETS}}
        for task, control in zip(tasks, auxiliary)
    ]


def target_values(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target in PRIMARY_TARGETS:
        return V1.target_values(rows, target)
    if target in CONTROL_TARGETS:
        return np.asarray([row[target] for row in rows], dtype=float)
    raise ValueError(target)


def group_key(row: dict[str, Any], pooled: bool) -> tuple[Any, ...]:
    return (row["world"], int(row["k"])) if pooled else (row["world"],)


def group_means(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    pooled: bool,
) -> dict[tuple[Any, ...], np.ndarray]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row, pooled)].append(index)
    return {
        key: np.asarray(values[indices], dtype=float).mean(axis=0)
        for key, indices in groups.items()
    }


def residualize(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    means: dict[tuple[Any, ...], np.ndarray],
    *,
    pooled: bool,
) -> np.ndarray:
    return np.asarray(
        [np.asarray(value, dtype=float) - means[group_key(row, pooled)]
         for value, row in zip(values, rows)],
        dtype=float,
    )


def add_means(
    residuals: np.ndarray,
    rows: list[dict[str, Any]],
    means: dict[tuple[Any, ...], np.ndarray],
    *,
    pooled: bool,
) -> np.ndarray:
    return np.asarray(
        [np.asarray(value, dtype=float) + means[group_key(row, pooled)]
         for value, row in zip(residuals, rows)],
        dtype=float,
    )


def feature_group_means(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    pooled: bool,
) -> dict[tuple[Any, ...], np.ndarray]:
    return group_means(features, rows, pooled=pooled)


def center_features(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    means: dict[tuple[Any, ...], np.ndarray],
    *,
    pooled: bool,
) -> np.ndarray:
    return residualize(features, rows, means, pooled=pooled)


def mean_align_features(
    cross_features: np.ndarray,
    cross_rows: list[dict[str, Any]],
    source_means: dict[tuple[Any, ...], np.ndarray],
    cross_means: dict[tuple[Any, ...], np.ndarray],
    *,
    pooled: bool,
) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(value, dtype=float)
            - cross_means[group_key(row, pooled)]
            + source_means[group_key(row, pooled)]
            for value, row in zip(cross_features, cross_rows)
        ],
        dtype=float,
    )


def _as_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[:, None] if array.ndim == 1 else array


def fair_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    pooled: bool,
) -> dict[str, Any]:
    prediction = _as_matrix(prediction)
    target = _as_matrix(target)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row, pooled)].append(index)
    by_group = {}
    for key, indices in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        y = target[indices]
        p = prediction[indices]
        error = y - p
        fair_denominator = float(np.sum(y**2))
        oracle_denominator = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
        centered_y = (y - y.mean(axis=0, keepdims=True)).reshape(-1)
        centered_p = (p - p.mean(axis=0, keepdims=True)).reshape(-1)
        correlation = (
            float(np.corrcoef(centered_y, centered_p)[0, 1])
            if np.std(centered_y) > 1e-12 and np.std(centered_p) > 1e-12
            else None
        )
        by_group["|".join(map(str, key))] = {
            "n": len(indices),
            "development_baseline_r2": (
                1.0 - float(np.sum(error**2)) / fair_denominator
                if fair_denominator > 1e-12 else None
            ),
            "oracle_test_mean_r2": (
                1.0 - float(np.sum(error**2)) / oracle_denominator
                if oracle_denominator > 1e-12 else None
            ),
            "centered_correlation": correlation,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "development_baseline_mae": float(np.mean(np.abs(y))),
        }
    finite_fair = [
        value["development_baseline_r2"]
        for value in by_group.values()
        if value["development_baseline_r2"] is not None
    ]
    finite_oracle = [
        value["oracle_test_mean_r2"]
        for value in by_group.values()
        if value["oracle_test_mean_r2"] is not None
    ]
    finite_corr = [
        value["centered_correlation"]
        for value in by_group.values()
        if value["centered_correlation"] is not None
    ]
    return {
        "n": len(target),
        "groups": len(by_group),
        "macro_development_baseline_r2": float(np.mean(finite_fair)),
        "macro_oracle_test_mean_r2": float(np.mean(finite_oracle)),
        "macro_centered_correlation": (
            float(np.mean(finite_corr)) if finite_corr else None
        ),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "mae": float(np.mean(np.abs(target - prediction))),
        "development_baseline_mae": float(np.mean(np.abs(target))),
        "mae_improvement_over_development_baseline": float(
            np.mean(np.abs(target)) - np.mean(np.abs(target - prediction))
        ),
        "by_group": by_group,
    }


def indices_for(
    tasks: list[dict[str, Any]],
    *,
    disclosure: str,
    ks: Iterable[int],
    split: str,
) -> np.ndarray:
    selected = set(int(value) for value in ks)
    return np.asarray(
        [
            index for index, row in enumerate(tasks)
            if row["disclosure"] == disclosure
            and int(row["k"]) in selected
            and row["split"] == split
        ],
        dtype=int,
    )


def cross_validated_alpha(
    *,
    train: np.ndarray,
    raw_target: np.ndarray,
    rows: list[dict[str, Any]],
    folds: np.ndarray,
    pooled: bool,
    nuisance_center_features: bool,
) -> tuple[float, float, dict[str, float]]:
    unique_folds = sorted(set(int(value) for value in folds))
    scores: dict[str, float] = {}
    predictions = {
        float(alpha): np.empty_like(raw_target, dtype=float) for alpha in ALPHAS
    }
    residual_targets = np.empty_like(raw_target, dtype=float)
    for fold in unique_folds:
        validation = folds == fold
        fitting = ~validation
        fit_rows = [row for row, keep in zip(rows, fitting) if keep]
        val_rows = [row for row, keep in zip(rows, validation) if keep]
        means = group_means(raw_target[fitting], fit_rows, pooled=pooled)
        fit_target = residualize(
            raw_target[fitting], fit_rows, means, pooled=pooled
        )
        val_target = residualize(
            raw_target[validation], val_rows, means, pooled=pooled
        )
        residual_targets[validation] = val_target
        fit_features = np.asarray(train[fitting], dtype=float)
        val_features = np.asarray(train[validation], dtype=float)
        if nuisance_center_features:
            feature_means = feature_group_means(fit_features, fit_rows, pooled=pooled)
            fit_features = center_features(
                fit_features, fit_rows, feature_means, pooled=pooled
            )
            val_features = center_features(
                val_features, val_rows, feature_means, pooled=pooled
            )
        design = V1.prepare_dual(fit_features, val_features)
        for alpha in ALPHAS:
            predictions[float(alpha)][validation] = V1.dual_predict(
                design, fit_target, float(alpha)
            )
    for alpha in ALPHAS:
        metrics = fair_metrics(
            predictions[float(alpha)], residual_targets, rows, pooled=pooled
        )
        scores[f"{alpha:g}"] = metrics["macro_development_baseline_r2"]
    best_alpha = max(
        (float(alpha) for alpha in ALPHAS),
        key=lambda alpha: (scores[f"{alpha:g}"], -alpha),
    )
    return best_alpha, scores[f"{best_alpha:g}"], scores


def permute_within_groups(
    target: np.ndarray,
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    pooled: bool,
) -> np.ndarray:
    permuted = np.array(target, copy=True)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row, pooled)].append(index)
    for indices in groups.values():
        indices_array = np.asarray(indices, dtype=int)
        permuted[indices_array] = target[rng.permutation(indices_array)]
    return permuted


def permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
    repeats: int,
    *,
    pooled: bool,
) -> dict[str, Any]:
    observed = fair_metrics(prediction, target, rows, pooled=pooled)[
        "macro_development_baseline_r2"
    ]
    rng = np.random.default_rng(PERMUTATION_SEED)
    scores = np.asarray(
        [
            fair_metrics(
                prediction,
                permute_within_groups(target, rows, rng, pooled=pooled),
                rows,
                pooled=pooled,
            )["macro_development_baseline_r2"]
            for _ in range(repeats)
        ]
    )
    return {
        "scheme": "held-out target permutation within development-baseline group",
        "repeats": repeats,
        "seed": PERMUTATION_SEED,
        "observed_score": observed,
        "null_mean": float(scores.mean()),
        "null_q025": float(np.quantile(scores, 0.025)),
        "null_q975": float(np.quantile(scores, 0.975)),
        "one_sided_p": float((1 + np.sum(scores >= observed)) / (len(scores) + 1)),
    }


def prediction_rows(
    *,
    endpoint: str,
    representation: str,
    condition: str,
    evaluation: str,
    rows: list[dict[str, Any]],
    raw_target: np.ndarray,
    residual_target: np.ndarray,
    residual_prediction: np.ndarray,
    raw_prediction: np.ndarray,
    baseline_means: dict[tuple[Any, ...], np.ndarray],
    pooled: bool,
) -> list[dict[str, Any]]:
    output = []
    for row, raw_y, residual_y, residual_p, raw_p in zip(
        rows, raw_target, residual_target, residual_prediction, raw_prediction
    ):
        baseline = baseline_means[group_key(row, pooled)]
        output.append(
            {
                "endpoint": endpoint,
                "representation": representation,
                "condition": condition,
                "evaluation": evaluation,
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "episode_id": row["episode_id"],
                "world": row["world"],
                "k": int(row["k"]),
                "disclosure": row["disclosure"],
                "raw_target": np.asarray(raw_y).tolist(),
                "development_baseline": np.asarray(baseline).tolist(),
                "residual_target": np.asarray(residual_y).tolist(),
                "residual_prediction": np.asarray(residual_p).tolist(),
                "raw_prediction": np.asarray(raw_p).tolist(),
            }
        )
    return output


def analyze_representation(
    *,
    name: str,
    features: np.ndarray,
    tasks: list[dict[str, Any]],
    targets: tuple[str, ...],
    permutation_repeats: int,
    layerwise: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if features.ndim == 2:
        features = features[:, None, :]
    layer_count = features.shape[1]
    result: dict[str, Any] = {
        "representation": name,
        "layer_count": layer_count,
        "conditions": {},
    }
    csv_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    condition_specs = [((k,), False) for k in K_VALUES] + [(K_VALUES, True)]

    for disclosure in DISCLOSURES:
        opposite = "undisclosed" if disclosure == "disclosed" else "disclosed"
        for ks, pooled in condition_specs:
            selected_targets = targets if not pooled else PRIMARY_TARGETS
            dev_indices = indices_for(
                tasks, disclosure=disclosure, ks=ks, split="dev"
            )
            test_indices = indices_for(
                tasks, disclosure=disclosure, ks=ks, split="test"
            )
            cross_dev_indices = indices_for(
                tasks, disclosure=opposite, ks=ks, split="dev"
            )
            cross_indices = indices_for(
                tasks, disclosure=opposite, ks=ks, split="test"
            )
            dev_rows = [tasks[index] for index in dev_indices]
            test_rows = [tasks[index] for index in test_indices]
            cross_dev_rows = [tasks[index] for index in cross_dev_indices]
            cross_rows = [tasks[index] for index in cross_indices]
            folds = np.asarray([row["dev_fold"] for row in dev_rows], dtype=int)
            for target_name in selected_targets:
                raw_dev_target = target_values(dev_rows, target_name)
                raw_test_target = target_values(test_rows, target_name)
                raw_cross_target = target_values(cross_rows, target_name)
                layers = []
                nuisance_center_features = pooled
                for layer_index in range(layer_count):
                    alpha, cv_score, alpha_scores = cross_validated_alpha(
                        train=features[dev_indices, layer_index, :],
                        raw_target=raw_dev_target,
                        rows=dev_rows,
                        folds=folds,
                        pooled=pooled,
                        nuisance_center_features=nuisance_center_features,
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
                            "train_disclosure": disclosure,
                            "evidence": "pooled" if pooled else f"k{ks[0]}",
                            "target": target_name,
                            "layer": layer_index,
                            "alpha": alpha,
                            "cv_macro_development_baseline_r2": cv_score,
                            "selected": False,
                        }
                    )
                selected_layer = max(
                    range(layer_count),
                    key=lambda layer: (layers[layer]["cv_score"], -layer),
                )
                selected = layers[selected_layer]
                for row in reversed(csv_rows):
                    if (
                        row["representation"] == name
                        and row["train_disclosure"] == disclosure
                        and row["evidence"] == ("pooled" if pooled else f"k{ks[0]}")
                        and row["target"] == target_name
                        and row["layer"] == selected_layer
                    ):
                        row["selected"] = True
                        break

                baseline_means = group_means(
                    raw_dev_target, dev_rows, pooled=pooled
                )
                dev_target = residualize(
                    raw_dev_target, dev_rows, baseline_means, pooled=pooled
                )
                test_target = residualize(
                    raw_test_target, test_rows, baseline_means, pooled=pooled
                )
                cross_target = residualize(
                    raw_cross_target, cross_rows, baseline_means, pooled=pooled
                )
                train_features = np.asarray(
                    features[dev_indices, selected_layer, :], dtype=float
                )
                same_features = np.asarray(
                    features[test_indices, selected_layer, :], dtype=float
                )
                cross_dev_features = np.asarray(
                    features[cross_dev_indices, selected_layer, :], dtype=float
                )
                cross_features = np.asarray(
                    features[cross_indices, selected_layer, :], dtype=float
                )
                source_feature_means = feature_group_means(
                    train_features, dev_rows, pooled=pooled
                )
                cross_feature_means = feature_group_means(
                    cross_dev_features, cross_dev_rows, pooled=pooled
                )
                aligned_cross_features = mean_align_features(
                    cross_features,
                    cross_rows,
                    source_feature_means,
                    cross_feature_means,
                    pooled=pooled,
                )
                if nuisance_center_features:
                    train_features = center_features(
                        train_features,
                        dev_rows,
                        source_feature_means,
                        pooled=pooled,
                    )
                    same_features = center_features(
                        same_features,
                        test_rows,
                        source_feature_means,
                        pooled=pooled,
                    )
                    cross_features = center_features(
                        cross_features,
                        cross_rows,
                        source_feature_means,
                        pooled=pooled,
                    )
                    aligned_cross_features = center_features(
                        aligned_cross_features,
                        cross_rows,
                        source_feature_means,
                        pooled=pooled,
                    )
                combined_eval = np.concatenate(
                    (same_features, cross_features, aligned_cross_features), axis=0
                )
                design = V1.prepare_dual(train_features, combined_eval)
                prediction = V1.dual_predict(
                    design, dev_target, selected["alpha"]
                )
                same_prediction = prediction[: len(test_indices)]
                cross_prediction = prediction[
                    len(test_indices) : len(test_indices) + len(cross_indices)
                ]
                aligned_prediction = prediction[
                    len(test_indices) + len(cross_indices) :
                ]
                condition = (
                    f"{disclosure}:pooled:{target_name}"
                    if pooled else f"{disclosure}:k{ks[0]}:{target_name}"
                )
                condition_result = {
                    "train_disclosure": disclosure,
                    "cross_disclosure": opposite,
                    "evidence": "pooled" if pooled else f"k{ks[0]}",
                    "pooled_evidence_depths": list(ks) if pooled else None,
                    "target": target_name,
                    "target_class": (
                        "primary" if target_name in PRIMARY_TARGETS else "numerical_control"
                    ),
                    "development_n": len(dev_rows),
                    "test_n": len(test_rows),
                    "selected_layer": selected_layer,
                    "selected_alpha": selected["alpha"],
                    "selected_cv_macro_development_baseline_r2": selected["cv_score"],
                    "target_centering": (
                        "fold-local development world x k means"
                        if pooled else "fold-local development world means"
                    ),
                    "feature_nuisance_centering": (
                        "fold-local development world x k means" if pooled else None
                    ),
                    "same_disclosure_test": fair_metrics(
                        same_prediction, test_target, test_rows, pooled=pooled
                    ),
                    "cross_disclosure_test": fair_metrics(
                        cross_prediction, cross_target, cross_rows, pooled=pooled
                    ),
                    "mean_aligned_cross_disclosure_test": fair_metrics(
                        aligned_prediction, cross_target, cross_rows, pooled=pooled
                    ),
                    "cross_alignment": (
                        "unlabeled development world x k mean translation"
                        if pooled else "unlabeled development world mean translation"
                    ),
                    "permutation": permutation_null(
                        same_prediction,
                        test_target,
                        test_rows,
                        permutation_repeats,
                        pooled=pooled,
                    ),
                    "layers": layers if layerwise else None,
                }
                result["conditions"][condition] = condition_result
                for evaluation, rows_, raw_y, residual_y, residual_p in (
                    ("same_disclosure", test_rows, raw_test_target, test_target, same_prediction),
                    ("cross_disclosure", cross_rows, raw_cross_target, cross_target, cross_prediction),
                    ("mean_aligned_cross_disclosure", cross_rows, raw_cross_target, cross_target, aligned_prediction),
                ):
                    raw_prediction = add_means(
                        residual_p, rows_, baseline_means, pooled=pooled
                    )
                    sealed_rows.extend(
                        prediction_rows(
                            endpoint="",
                            representation=name,
                            condition=condition,
                            evaluation=evaluation,
                            rows=rows_,
                            raw_target=raw_y,
                            residual_target=residual_y,
                            residual_prediction=residual_p,
                            raw_prediction=raw_prediction,
                            baseline_means=baseline_means,
                            pooled=pooled,
                        )
                    )
    return result, csv_rows, sealed_rows


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--auxiliary-targets", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--status", default="probe_v2_complete")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="skip numerical controls for a fast primary-analysis pass",
    )
    args = parser.parse_args()

    tasks_path = args.task_dir / "tasks.jsonl"
    tasks = read_jsonl(tasks_path)
    task_manifest = json.loads(
        (args.task_dir / "task_manifest.json").read_text(encoding="utf-8")
    )
    if file_sha256(tasks_path) != task_manifest["tasks_sha256"]:
        raise ValueError("tasks hash mismatch")
    tasks = attach_auxiliary(tasks, args.auxiliary_targets)
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

    targets = PRIMARY_TARGETS if args.primary_only else PRIMARY_TARGETS + CONTROL_TARGETS
    representations = {}
    csv_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    primary, rows, predictions = analyze_representation(
        name="final_input_token_layerwise",
        features=last_token,
        tasks=tasks,
        targets=targets,
        permutation_repeats=args.permutations,
        layerwise=True,
    )
    representations[primary["representation"]] = primary
    csv_rows.extend(rows)
    sealed_rows.extend(predictions)
    for control_index, name in enumerate(("mean_embedding", "mean_final_layer")):
        control, rows, predictions = analyze_representation(
            name=name,
            features=mean_control[:, control_index, :],
            tasks=tasks,
            targets=PRIMARY_TARGETS,
            permutation_repeats=args.permutations,
            layerwise=False,
        )
        representations[name] = control
        csv_rows.extend(rows)
        sealed_rows.extend(predictions)

    endpoint = extraction_manifest["endpoint_label"]
    for row in sealed_rows:
        row["endpoint"] = endpoint
    result = clean_json(
        {
            "study": "exp3_mechanism_probe_v2",
            "source_study": task_manifest["study"],
            "status": args.status,
            "model": extraction_manifest["model"],
            "model_label": extraction_manifest["model_label"],
            "endpoint_label": endpoint,
            "tasks_sha256": file_sha256(tasks_path),
            "auxiliary_targets_sha256": file_sha256(args.auxiliary_targets),
            "extraction_manifest_sha256": file_sha256(
                args.run_dir / "extraction_manifest.json"
            ),
            "features_sha256": extraction_manifest["features_sha256"],
            "primary_targets": list(PRIMARY_TARGETS),
            "numerical_controls": [] if args.primary_only else list(CONTROL_TARGETS),
            "primary_score": (
                "macro held-out R2 relative to development-estimated world or "
                "world x evidence-depth means"
            ),
            "secondary_score": "macro R2 relative to sealed-test oracle group means",
            "representations": representations,
        }
    )
    result_path = args.run_dir / "probe_results_v2.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = args.run_dir / "layerwise_results_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    predictions_path = args.run_dir / "sealed_predictions_v2.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(clean_json(row), sort_keys=True) + "\n" for row in sealed_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "probe_results_v2": str(result_path),
                "layerwise_results_v2": str(csv_path),
                "sealed_predictions_v2": str(predictions_path),
                "conditions": len(primary["conditions"]),
                "prediction_rows": len(sealed_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
