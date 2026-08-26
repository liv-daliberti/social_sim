#!/usr/bin/env python3
"""Fit leakage-safe layer-wise ridge probes to saved Qwen hidden states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RUN_DIR = (
    ROOT
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_8b_toy_v1"
)
ARMS = ("abc_context", "abc_no_context", "abc_wrong_context")
TARGETS = ("regime", "slope", "residual_slope")
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
    return DualDesign(
        train_kernel=train_z @ train_z.T,
        eval_kernel=eval_z @ train_z.T,
    )


def dual_predict(design: DualDesign, target: np.ndarray, alpha: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    center = float(target.mean())
    system = design.train_kernel + float(alpha) * np.eye(len(target))
    coefficients = np.linalg.solve(system, target - center)
    return design.eval_kernel @ coefficients + center


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(labels, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(prediction) < 2 or prediction.std() < 1e-12 or target.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    residual = target - prediction
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator <= 0 else 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
        "pearson": pearson(prediction, target),
    }


def classification_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=bool)
    return {
        "accuracy": float(np.mean((scores >= 0.0) == target)),
        "roc_auc": roc_auc(scores, target),
        "score_pearson": pearson(scores, target.astype(float)),
    }


def target_values(rows: list[dict[str, Any]], target_name: str) -> np.ndarray:
    if target_name == "regime":
        return np.asarray(
            [1.0 if row["target_strong"] else -1.0 for row in rows],
            dtype=float,
        )
    if target_name == "slope":
        return np.asarray([row["target_slope"] for row in rows], dtype=float)
    if target_name == "residual_slope":
        return np.asarray([row["residual_slope"] for row in rows], dtype=float)
    raise ValueError(target_name)


def score_for_selection(
    prediction: np.ndarray,
    target: np.ndarray,
    target_name: str,
) -> float:
    if target_name == "regime":
        return classification_metrics(prediction, target > 0)["roc_auc"]
    return regression_metrics(prediction, target)["r2"]


def cross_validated_alpha(
    *,
    train: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    target_name: str,
    alphas: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    unique_folds = sorted(set(int(value) for value in folds))
    designs: list[tuple[np.ndarray, DualDesign]] = []
    for fold in unique_folds:
        validation = folds == fold
        fitting = ~validation
        designs.append(
            (
                validation,
                prepare_dual(train[fitting], train[validation]),
            )
        )

    scores: dict[str, float] = {}
    for alpha in alphas:
        predictions = np.empty(len(target), dtype=float)
        for fold, (validation, design) in zip(unique_folds, designs):
            fitting = folds != fold
            predictions[validation] = dual_predict(
                design,
                target[fitting],
                float(alpha),
            )
        scores[f"{alpha:g}"] = score_for_selection(
            predictions,
            target,
            target_name,
        )
    best_alpha = max(
        (float(alpha) for alpha in alphas),
        key=lambda alpha: (
            -float("inf")
            if not np.isfinite(scores[f"{alpha:g}"])
            else scores[f"{alpha:g}"],
            -alpha,
        ),
    )
    return best_alpha, scores[f"{best_alpha:g}"], scores


def evaluate_predictions(
    prediction: np.ndarray,
    rows: list[dict[str, Any]],
    target_name: str,
) -> dict[str, Any]:
    target = target_values(rows, target_name)
    if target_name == "regime":
        result: dict[str, Any] = {
            "true_target": classification_metrics(prediction, target > 0)
        }
        cue_mask = np.asarray([row["cue_strong"] is not None for row in rows])
        if cue_mask.any():
            cue = np.asarray(
                [bool(row["cue_strong"]) for row in rows if row["cue_strong"] is not None]
            )
            result["cue_target"] = classification_metrics(prediction[cue_mask], cue)
        return result

    result = {"true_target": regression_metrics(prediction, target)}
    if target_name == "slope":
        cue_mask = np.asarray([row["cue_strong"] is not None for row in rows])
        if cue_mask.any():
            cue_slope = np.asarray(
                [
                    0.90 if row["cue_strong"] else 0.25
                    for row in rows
                    if row["cue_strong"] is not None
                ],
                dtype=float,
            )
            result["cue_target"] = regression_metrics(
                prediction[cue_mask],
                cue_slope,
            )
    return result


def analytic_baselines(
    rows: list[dict[str, Any]],
    target_name: str,
) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    if target_name == "regime":
        cue_scores = np.asarray(
            [
                0.0 if row["cue_strong"] is None else (1.0 if row["cue_strong"] else -1.0)
                for row in rows
            ]
        )
        baselines["cue_only"] = evaluate_predictions(cue_scores, rows, target_name)
        visible = np.asarray(
            [
                float("nan")
                if row["visible_target_ols"] is None
                else float(row["visible_target_ols"]) - 0.575
                for row in rows
            ]
        )
    elif target_name == "slope":
        cue_scores = np.asarray(
            [
                0.575
                if row["cue_strong"] is None
                else (0.90 if row["cue_strong"] else 0.25)
                for row in rows
            ]
        )
        baselines["cue_only"] = evaluate_predictions(cue_scores, rows, target_name)
        visible = np.asarray(
            [
                float("nan")
                if row["visible_target_ols"] is None
                else float(row["visible_target_ols"])
                for row in rows
            ]
        )
    else:
        zeros = np.zeros(len(rows), dtype=float)
        baselines["zero_residual"] = evaluate_predictions(zeros, rows, target_name)
        visible = np.asarray(
            [
                float("nan")
                if row["visible_target_ols"] is None
                else float(row["visible_target_ols"]) - float(row["regime_mean_slope"])
                for row in rows
            ]
        )
    usable = np.isfinite(visible)
    if usable.any():
        baselines["visible_city_c_ols"] = evaluate_predictions(
            visible[usable],
            [row for row, keep in zip(rows, usable) if keep],
            target_name,
        )
    return baselines


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


def behavior_results(
    behavior_path: Path,
    task_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not behavior_path.exists():
        return {"status": "missing"}
    behavior = read_jsonl(behavior_path)
    results: dict[str, Any] = {"status": "complete", "conditions": {}}
    for c_cases in (0, 4):
        for arm in ARMS:
            condition = [
                row
                for row in behavior
                if row["c_cases"] == c_cases and row["arm"] == arm
            ]
            tasks = [task_by_id[row["sample_id"]] for row in condition]
            parsed = np.asarray([row["implied_slope"] is not None for row in condition])
            key = f"k{c_cases}:{arm}"
            item: dict[str, Any] = {
                "n": len(condition),
                "parsed": int(parsed.sum()),
                "parse_rate": float(parsed.mean()) if len(parsed) else float("nan"),
            }
            if parsed.any():
                kept_rows = [row for row, keep in zip(condition, parsed) if keep]
                kept_tasks = [row for row, keep in zip(tasks, parsed) if keep]
                slopes = np.asarray([row["implied_slope"] for row in kept_rows], dtype=float)
                polls = np.asarray([row["predicted_poll"] for row in kept_rows], dtype=float)
                gold_polls = np.asarray(
                    [row["gold_expected_poll"] for row in kept_tasks],
                    dtype=float,
                )
                item["poll"] = regression_metrics(polls, gold_polls)
                item["slope"] = evaluate_predictions(slopes, kept_tasks, "slope")
                item["regime"] = evaluate_predictions(
                    slopes - 0.575,
                    kept_tasks,
                    "regime",
                )
            results["conditions"][key] = item
    return results


def permutation_null(
    *,
    train: np.ndarray,
    test: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    alpha: float,
    target_name: str,
    repeats: int,
) -> dict[str, Any]:
    design = prepare_dual(train, test)
    rng = np.random.default_rng(PERMUTATION_SEED)
    observed_prediction = dual_predict(design, train_target, alpha)
    observed = score_for_selection(observed_prediction, test_target, target_name)
    null_scores = []
    for _ in range(repeats):
        shuffled_test_target = rng.permutation(test_target)
        null_scores.append(
            score_for_selection(
                observed_prediction, shuffled_test_target, target_name
            )
        )
    finite = np.asarray([score for score in null_scores if np.isfinite(score)], dtype=float)
    p_value = (
        float("nan")
        if not len(finite) or not np.isfinite(observed)
        else float((1 + np.sum(finite >= observed)) / (len(finite) + 1))
    )
    return {
        "scheme": (
            "held-out target-label permutation with the development-selected "
            "layer, ridge strength, and fitted probe held fixed"
        ),
        "repeats": repeats,
        "seed": PERMUTATION_SEED,
        "observed_score": observed,
        "null_mean": float(finite.mean()) if len(finite) else float("nan"),
        "null_q025": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "null_q975": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "one_sided_p": p_value,
    }


def analyze_representation(
    *,
    name: str,
    features: np.ndarray,
    tasks: list[dict[str, Any]],
    alphas: np.ndarray,
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
    selected_rows: list[dict[str, Any]] = []

    for c_cases in (0, 4):
        dev_indices = np.asarray(
            [
                index
                for index, row in enumerate(tasks)
                if row["split"] == "dev"
                and row["arm"] == "abc_context"
                and row["c_cases"] == c_cases
            ],
            dtype=int,
        )
        test_indices = np.asarray(
            [
                index
                for index, row in enumerate(tasks)
                if row["split"] == "test" and row["c_cases"] == c_cases
            ],
            dtype=int,
        )
        dev_rows = [tasks[index] for index in dev_indices]
        test_rows = [tasks[index] for index in test_indices]
        folds = np.asarray([row["cv_fold"] for row in dev_rows], dtype=int)
        if set(folds.tolist()) != {0, 1, 2, 3}:
            raise ValueError("development folds are incomplete")
        test_arm_indices = {
            arm: np.asarray(
                [index for index, row in enumerate(test_rows) if row["arm"] == arm],
                dtype=int,
            )
            for arm in ARMS
        }

        per_target: dict[str, list[dict[str, Any]]] = {
            target: [] for target in TARGETS
        }
        for layer in range(layer_count):
            dev_x = features[dev_indices, layer, :]
            test_x = features[test_indices, layer, :]
            full_design = prepare_dual(dev_x, test_x)
            for target_name in TARGETS:
                dev_y = target_values(dev_rows, target_name)
                alpha, cv_score, alpha_scores = cross_validated_alpha(
                    train=dev_x,
                    target=dev_y,
                    folds=folds,
                    target_name=target_name,
                    alphas=alphas,
                )
                prediction = dual_predict(full_design, dev_y, alpha)
                arm_metrics: dict[str, Any] = {}
                for arm, relative_indices in test_arm_indices.items():
                    arm_rows = [test_rows[index] for index in relative_indices]
                    arm_prediction = prediction[relative_indices]
                    arm_metrics[arm] = {
                        "probe": evaluate_predictions(
                            arm_prediction,
                            arm_rows,
                            target_name,
                        ),
                        "baselines": analytic_baselines(arm_rows, target_name),
                    }
                layer_result = {
                    "layer": layer,
                    "alpha": alpha,
                    "cv_score": cv_score,
                    "alpha_cv_scores": alpha_scores,
                    "test": arm_metrics,
                }
                per_target[target_name].append(layer_result)
                csv_rows.append(
                    {
                        "representation": name,
                        "c_cases": c_cases,
                        "target": target_name,
                        "layer": layer,
                        "alpha": alpha,
                        "cv_score": cv_score,
                        "correct_primary": (
                            arm_metrics["abc_context"]["probe"]["true_target"][
                                "roc_auc" if target_name == "regime" else "r2"
                            ]
                        ),
                        "no_context_primary": (
                            arm_metrics["abc_no_context"]["probe"]["true_target"][
                                "roc_auc" if target_name == "regime" else "r2"
                            ]
                        ),
                        "wrong_context_primary": (
                            arm_metrics["abc_wrong_context"]["probe"]["true_target"][
                                "roc_auc" if target_name == "regime" else "r2"
                            ]
                        ),
                    }
                )

        for target_name, layers in per_target.items():
            best = max(
                layers,
                key=lambda row: (
                    -float("inf")
                    if not np.isfinite(row["cv_score"])
                    else row["cv_score"],
                    -row["layer"],
                ),
            )
            best_layer = int(best["layer"])
            dev_x = features[dev_indices, best_layer, :]
            test_x = features[test_indices, best_layer, :]
            dev_y = target_values(dev_rows, target_name)
            test_y = target_values(test_rows, target_name)
            prediction = dual_predict(
                prepare_dual(dev_x, test_x),
                dev_y,
                float(best["alpha"]),
            )
            correct_relative = test_arm_indices["abc_context"]
            null = permutation_null(
                train=dev_x,
                test=test_x[correct_relative],
                train_target=dev_y,
                test_target=test_y[correct_relative],
                alpha=float(best["alpha"]),
                target_name=target_name,
                repeats=permutation_repeats,
            )
            for row, predicted in zip(test_rows, prediction):
                selected_rows.append(
                    {
                        "representation": name,
                        "c_cases": c_cases,
                        "target": target_name,
                        "selected_layer": best_layer,
                        "alpha": float(best["alpha"]),
                        "sample_id": row["sample_id"],
                        "episode": row["episode"],
                        "arm": row["arm"],
                        "prediction": float(predicted),
                        "target": float(target_values([row], target_name)[0]),
                        "cue_strong": row["cue_strong"],
                    }
                )
            result["conditions"][f"k{c_cases}:{target_name}"] = {
                "selected_by": "maximum development cross-validation score",
                "selected_layer": best_layer,
                "selected_alpha": float(best["alpha"]),
                "development_cv_score": float(best["cv_score"]),
                "selected_test": best["test"],
                "permutation_null_on_correct_context_test": null,
                "layers": layers if layerwise else None,
            }
    return result, csv_rows, selected_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--study", default="qwen3_8b_toy_v1")
    parser.add_argument("--status", default="toy_complete")
    args = parser.parse_args()

    tasks_path = args.run_dir / "tasks.jsonl"
    features_path = args.run_dir / "hidden_states.npz"
    tasks = read_jsonl(tasks_path)
    with np.load(features_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        last_token = payload["last_token"].astype(np.float32)
        mean_control = payload["mean_control"].astype(np.float32)
        mean_control_layers = payload["mean_control_layer_indices"].astype(int).tolist()
    expected_ids = [row["sample_id"] for row in tasks]
    if sample_ids != expected_ids:
        raise ValueError("feature sample order does not match tasks.jsonl")

    primary, csv_rows, selected_rows = analyze_representation(
        name="last_input_token",
        features=last_token,
        tasks=tasks,
        alphas=ALPHAS,
        permutation_repeats=args.permutations,
        layerwise=True,
    )
    controls: dict[str, Any] = {}
    for control_index, source_layer in enumerate(mean_control_layers):
        name = f"mean_pool_source_layer_{source_layer}"
        control, control_csv, control_selected = analyze_representation(
            name=name,
            features=mean_control[:, control_index, :],
            tasks=tasks,
            alphas=ALPHAS,
            permutation_repeats=args.permutations,
            layerwise=False,
        )
        controls[name] = control
        csv_rows.extend(control_csv)
        selected_rows.extend(control_selected)

    task_by_id = {row["sample_id"]: row for row in tasks}
    results = {
        "study": args.study,
        "status": args.status,
        "claim_scope": (
            "correlational linear decodability; not evidence of causal mediation"
        ),
        "targets": list(TARGETS),
        "training_scope": (
            "development abc_context only; separate probes at k=0 and k=4"
        ),
        "selection": (
            "ridge strength and one reported layer selected by four-fold "
            "cell-balanced development CV before test evaluation"
        ),
        "alphas": ALPHAS.tolist(),
        "permutations": args.permutations,
        "tasks_sha256": file_sha256(tasks_path),
        "features_sha256": file_sha256(features_path),
        "primary": primary,
        "pooling_controls": controls,
        "behavior": behavior_results(
            args.run_dir / "behavior_test.jsonl",
            task_by_id,
        ),
    }
    cleaned = clean_json(results)
    results_path = args.run_dir / "probe_results.json"
    results_path.write_text(
        json.dumps(cleaned, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.run_dir / "layerwise_results.csv", clean_json(csv_rows))
    predictions_content = "".join(
        json.dumps(clean_json(row), sort_keys=True) + "\n"
        for row in selected_rows
    )
    (args.run_dir / "selected_predictions.jsonl").write_text(
        predictions_content,
        encoding="utf-8",
    )

    summary = {}
    for key, item in cleaned["primary"]["conditions"].items():
        target_name = key.split(":", 1)[1]
        metric = "roc_auc" if target_name == "regime" else "r2"
        summary[key] = {
            "layer": item["selected_layer"],
            "cv": item["development_cv_score"],
            "correct": item["selected_test"]["abc_context"]["probe"]["true_target"][metric],
            "no_context": item["selected_test"]["abc_no_context"]["probe"]["true_target"][metric],
            "wrong_context": item["selected_test"]["abc_wrong_context"]["probe"]["true_target"][metric],
            "permutation_p": item["permutation_null_on_correct_context_test"]["one_sided_p"],
        }
    print(
        json.dumps(
            {
                "results": str(results_path),
                "summary": summary,
                "behavior": cleaned["behavior"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

