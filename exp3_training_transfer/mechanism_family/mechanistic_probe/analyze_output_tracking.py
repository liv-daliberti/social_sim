#!/usr/bin/env python3
"""Formal output-level positive control for the Exp3 mechanism probe.

Generated impulse responses are evaluated on the identical development/test
episode split as the hidden-state probe.  Development rows alone define the
world baseline and the gain calibration.  Repeated stochastic draws are
averaged within task before any metric or bootstrap is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
DEFAULT_TASKS = HERE / "data" / "tasks.jsonl"
DEFAULT_AGGREGATE = FAMILY / "reports" / "c3_mechanism_full_aggregate.json"
DEFAULT_OUTPUT = HERE / "runs" / "output_tracking_v2.json"
DEFAULT_PREDICTIONS = HERE / "runs" / "output_tracking_predictions_v2.jsonl"
BOOTSTRAP_SEED = 20260825


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("exp3_output_probe_v1", HERE / "analyze_probe.py")
V2 = load_module("exp3_output_probe_v2", HERE / "analyze_probe_v2.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_files(aggregate: Path, model: str, seeds: set[int]) -> list[Path]:
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    paths = []
    for value in payload["score_files"]:
        if not value.endswith("/stochastic_n5.scores.jsonl") or "/debug_" in value:
            continue
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            first = json.loads(next(line for line in handle if line.strip()))
        if first["model"] != model:
            continue
        seed = first.get("seed")
        if first["arm"] == "base" or seed is None or int(seed) in seeds:
            paths.append(path)
    return sorted(paths)


def task_draw_means(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = None
    total = parsed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = {
                "model": row["model"],
                "disclosure": row["disclosure"],
                "arm": row["arm"],
                "seed": row.get("seed"),
            }
            total += 1
            if row.get("predicted_response") is not None:
                parsed += 1
                grouped[row["task_id"]].append(
                    np.asarray(row["predicted_response"], dtype=float)
                )
    if metadata is None:
        raise ValueError(f"empty score file: {path}")
    means = {task_id: np.mean(values, axis=0) for task_id, values in grouped.items()}
    return metadata, means, parsed / total


def macro_bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    rows: list[dict[str, Any]],
    repetitions: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups: dict[str, np.ndarray] = {}
    worlds = sorted({row["world"] for row in rows})
    for world in worlds:
        groups[world] = np.asarray(
            [index for index, row in enumerate(rows) if row["world"] == world],
            dtype=int,
        )
    draws = defaultdict(list)
    for _ in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()]
        )
        sampled_rows = [rows[index] for index in sampled]
        metrics = V2.fair_metrics(
            prediction[sampled], target[sampled], sampled_rows, pooled=False
        )
        for key in (
            "macro_development_baseline_r2",
            "macro_centered_correlation",
            "mae_improvement_over_development_baseline",
        ):
            if metrics[key] is not None:
                draws[key].append(float(metrics[key]))
    return {
        key: [float(value) for value in np.quantile(values, (0.025, 0.975))]
        for key, values in draws.items()
    }


def condition_analysis(
    *,
    metadata: dict[str, Any],
    output_means: dict[str, np.ndarray],
    tasks: list[dict[str, Any]],
    k: int,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disclosure = metadata["disclosure"]
    selected = [
        row for row in tasks
        if row["disclosure"] == disclosure and int(row["k"]) == k
        and row["task_id"] in output_means
    ]
    dev_rows = [row for row in selected if row["split"] == "dev"]
    test_rows = [row for row in selected if row["split"] == "test"]
    if len(dev_rows) != 180 or len(test_rows) != 60:
        raise ValueError(
            f"{metadata}/{k}: expected 180 development and 60 test outputs, "
            f"found {len(dev_rows)} and {len(test_rows)}"
        )

    def output_residual(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                output_means[row["task_id"]] - np.asarray(row["prior_response"], dtype=float)
                for row in rows
            ]
        ) / 4.0

    dev_truth = V1.target_values(dev_rows, "response_residual")
    test_truth = V1.target_values(test_rows, "response_residual")
    dev_prediction = output_residual(dev_rows)
    test_prediction = output_residual(test_rows)
    response_means = V2.group_means(dev_truth, dev_rows, pooled=False)
    dev_response_target = V2.residualize(
        dev_truth, dev_rows, response_means, pooled=False
    )
    test_response_target = V2.residualize(
        test_truth, test_rows, response_means, pooled=False
    )
    dev_response_prediction = V2.residualize(
        dev_prediction, dev_rows, response_means, pooled=False
    )
    test_response_prediction = V2.residualize(
        test_prediction, test_rows, response_means, pooled=False
    )
    response_metrics = V2.fair_metrics(
        test_response_prediction, test_response_target, test_rows, pooled=False
    )
    response_metrics["bootstrap_ci95"] = macro_bootstrap(
        test_response_prediction,
        test_response_target,
        test_rows,
        repetitions,
    )

    dev_gain = V1.target_values(dev_rows, "gain")
    test_gain = V1.target_values(test_rows, "gain")
    folds = np.asarray([row["dev_fold"] for row in dev_rows], dtype=int)
    alpha, cv_score, alpha_scores = V2.cross_validated_alpha(
        train=dev_prediction,
        raw_target=dev_gain,
        rows=dev_rows,
        folds=folds,
        pooled=False,
        nuisance_center_features=False,
    )
    gain_means = V2.group_means(dev_gain, dev_rows, pooled=False)
    dev_gain_target = V2.residualize(dev_gain, dev_rows, gain_means, pooled=False)
    test_gain_target = V2.residualize(test_gain, test_rows, gain_means, pooled=False)
    design = V1.prepare_dual(dev_prediction, test_prediction)
    gain_prediction = V1.dual_predict(design, dev_gain_target, alpha)
    gain_metrics = V2.fair_metrics(
        gain_prediction, test_gain_target, test_rows, pooled=False
    )
    gain_metrics.update(
        {
            "selected_alpha": alpha,
            "development_cv_r2": cv_score,
            "development_alpha_scores": alpha_scores,
            "bootstrap_ci95": macro_bootstrap(
                gain_prediction, test_gain_target, test_rows, repetitions
            ),
        }
    )
    prediction_rows = []
    for row, truth, predicted, gain_y, gain_p in zip(
        test_rows,
        test_response_target,
        test_response_prediction,
        test_gain_target,
        gain_prediction,
    ):
        prediction_rows.append(
            {
                **metadata,
                "task_id": row["task_id"],
                "episode_id": row["episode_id"],
                "world": row["world"],
                "k": k,
                "response_residual_target": np.asarray(truth).tolist(),
                "response_residual_prediction": np.asarray(predicted).tolist(),
                "gain_residual_target": float(gain_y),
                "gain_residual_prediction": float(gain_p),
            }
        )
    return {
        **metadata,
        "k": k,
        "development_n": len(dev_rows),
        "test_n": len(test_rows),
        "response_tracking": response_metrics,
        "gain_calibration": gain_metrics,
    }, prediction_rows


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    files = score_files(args.aggregate, args.model, set(args.seeds))
    if not files:
        raise FileNotFoundError("no matching stochastic score files")
    conditions = []
    predictions = []
    sources = {}
    for path in files:
        metadata, means, coverage = task_draw_means(path)
        metadata["parse_coverage"] = coverage
        metadata["score_file"] = str(path)
        sources[str(path)] = file_sha256(path)
        for k in (3, 6, 9):
            condition, rows = condition_analysis(
                metadata=metadata,
                output_means=means,
                tasks=tasks,
                k=k,
                repetitions=args.bootstrap,
            )
            conditions.append(clean(condition))
            predictions.extend(clean(rows))
    result = {
        "study": "exp3_output_episode_tracking_v2",
        "status": "complete",
        "model": args.model,
        "training_seeds": args.seeds,
        "tasks_sha256": file_sha256(args.tasks),
        "aggregate_sha256": file_sha256(args.aggregate),
        "score_file_sha256": sources,
        "variance_unit": "episode; stochastic draws averaged within task",
        "development_baseline": "development response/gain mean within world",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "predictions": str(args.predictions),
                "score_files": len(files),
                "conditions": len(conditions),
                "prediction_rows": len(predictions),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
