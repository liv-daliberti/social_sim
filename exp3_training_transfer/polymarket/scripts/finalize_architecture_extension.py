#!/usr/bin/env python3
"""Fail-closed finalizer for the post-hoc Exp4 local architecture extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


POLY = Path(__file__).resolve().parents[1]
REPORTS = POLY / "reports"
RUNS = POLY / "runs"
REGISTRATION = (
    RUNS / "exp4_architecture_provider_sweep_registration_20260825T203304Z.json"
)
EXECUTION = (
    RUNS
    / "exp4_architecture_provider_sweep_all_local_seeds_started_20260825T212445Z.json"
)
HOLDOUT = POLY / "data" / "exp4_scale_registered" / "test.tasks.jsonl"
OUTPUT = REPORTS / "exp4_architecture_extension_latest.summary.json"
MODEL_KEYS = ("qwen3_1_7b", "llama3_2_3b")
SEEDS = ("42", "43", "44")
EXPECTED_ENDPOINTS = {"base", "seed_42", "seed_43", "seed_44"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc
    return rows


def resolve_report(model_key: str, evaluation_job_id: str) -> tuple[Path, Path]:
    prefix = REPORTS / (f"exp4_scale_{model_key}_stochastic_j{evaluation_job_id}")
    summary = prefix.with_suffix(".summary.json")
    raw = prefix.with_suffix(".jsonl")
    if not summary.is_file() or not raw.is_file():
        raise RuntimeError(
            f"missing completed {model_key} evaluation for job {evaluation_job_id}"
        )
    return summary, raw


def validate_registration(
    registration_path: Path, execution_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    registration = load_json(registration_path)
    execution = load_json(execution_path)
    if registration.get("status") != "registered":
        raise RuntimeError("architecture sweep registration is not frozen")
    if registration.get("protocol_version") != ("exp4_architecture_provider_sweep_v1"):
        raise RuntimeError("architecture sweep protocol drifted")
    if execution.get("status") != "all_six_local_training_seeds_running":
        raise RuntimeError("local execution ledger is not the all-seeds roster")
    if execution.get("protocol_version") != registration["protocol_version"]:
        raise RuntimeError("execution protocol does not match registration")
    if execution["scheduler"].get("scientific_settings_changed") is not False:
        raise RuntimeError("execution ledger records a scientific change")
    if execution.get("hosted_holdout_calls_started") is not False:
        raise RuntimeError("execution ledger unexpectedly opened hosted holdout")
    return registration, execution


def validate_task_universe(raw_path: Path) -> int:
    expected = read_jsonl(HOLDOUT)
    observed = read_jsonl(raw_path)
    expected_ids = [row["task_id"] for row in expected]
    observed_ids = [row.get("task_id") for row in observed]
    if len(expected_ids) != 318 or len(observed_ids) != 318:
        raise RuntimeError("architecture evaluation does not contain 318 tasks")
    if observed_ids != expected_ids:
        raise RuntimeError("architecture evaluation task order or identity drifted")
    if len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError("architecture evaluation contains duplicate task IDs")
    return len(observed_ids)


def validate_adapter_lineage(
    model_key: str,
    summary: dict[str, Any],
    execution_model: dict[str, Any],
) -> dict[str, dict[str, str]]:
    lineage: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        job_id = execution_model[seed]["job_id"]
        adapter_path = Path(summary["adapter_paths"][seed])
        if f"_j{job_id}" not in str(adapter_path):
            raise RuntimeError(
                f"{model_key} seed {seed} adapter is not from job {job_id}"
            )
        weights = adapter_path / "adapter_model.safetensors"
        if not weights.is_file():
            raise RuntimeError(f"{model_key} seed {seed} adapter weights are missing")
        weights_hash = sha256(weights)
        if summary["adapter_sha256"][seed] != weights_hash:
            raise RuntimeError(f"{model_key} seed {seed} adapter hash drifted")
        lineage[seed] = {
            "job_id": job_id,
            "path": str(adapter_path),
            "weights_sha256": weights_hash,
        }
    return lineage


def validate_endpoint(name: str, endpoint: dict[str, Any]) -> None:
    for key in ("brier", "draw_parse_coverage", "complete_task_parse_coverage"):
        value = endpoint.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"{name} has invalid {key}")
    if not 0.0 <= endpoint["brier"] <= 1.0:
        raise RuntimeError(f"{name} Brier score is outside [0, 1]")
    for key in ("draw_parse_coverage", "complete_task_parse_coverage"):
        if not 0.0 <= endpoint[key] <= 1.0:
            raise RuntimeError(f"{name} {key} is outside [0, 1]")


def validate_model(
    model_key: str,
    registration: dict[str, Any],
    execution: dict[str, Any],
    summary_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    registered = registration["local_models"][model_key]
    execution_model = execution[model_key]
    evaluation_job_id = execution_model["evaluation"]["job_id"]

    if summary.get("protocol_version") != "exp4_qwen_scale_v1":
        raise RuntimeError(f"{model_key} parent scale protocol drifted")
    if summary.get("model_key") != model_key:
        raise RuntimeError(f"{model_key} summary identity drifted")
    if summary.get("model") != registered["snapshot"]:
        raise RuntimeError(f"{model_key} checkpoint drifted")
    if summary.get("decoding") != registration["locked_evaluation"]["decoding"]:
        raise RuntimeError(f"{model_key} decoder drifted")
    holdout = summary.get("holdout", {})
    frozen = registration["frozen_data"]
    if holdout.get("n") != 318:
        raise RuntimeError(f"{model_key} holdout count drifted")
    if holdout.get("test_tasks_sha256") != frozen["holdout_tasks_sha256"]:
        raise RuntimeError(f"{model_key} holdout hash drifted")
    if holdout.get("manifest_sha256") != frozen["holdout_manifest_sha256"]:
        raise RuntimeError(f"{model_key} holdout manifest drifted")
    if summary.get("analysis_code_sha256") != (
        registration["frozen_code"]["evaluator_sha256"]
    ):
        raise RuntimeError(f"{model_key} evaluator hash drifted")
    if set(summary.get("models", {})) != EXPECTED_ENDPOINTS:
        raise RuntimeError(f"{model_key} endpoint roster drifted")
    for endpoint_name, endpoint in summary["models"].items():
        validate_endpoint(f"{model_key}:{endpoint_name}", endpoint)
    if set(summary.get("adapter_paths", {})) != set(SEEDS):
        raise RuntimeError(f"{model_key} adapter roster drifted")
    if set(summary.get("adapter_sha256", {})) != set(SEEDS):
        raise RuntimeError(f"{model_key} adapter hash roster drifted")

    raw_rows = validate_task_universe(raw_path)
    lineage = validate_adapter_lineage(model_key, summary, execution_model)
    trained = [float(summary["models"][f"seed_{seed}"]["brier"]) for seed in SEEDS]
    trained_mean = sum(trained) / len(trained)
    market_brier = float(summary["baselines"]["market"]["brier"])
    comparison = summary["comparisons_brier"]
    for required in (
        "trained_seed_mean_minus_base",
        "trained_seed_mean_minus_market",
    ):
        if required not in comparison:
            raise RuntimeError(f"{model_key} is missing comparison {required}")

    return {
        "checkpoint": registered["checkpoint"],
        "revision": registered["revision"],
        "figure_role": registered["figure_role"],
        "evaluation_job_id": evaluation_job_id,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "raw": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "raw_rows": raw_rows,
        "adapter_lineage": lineage,
        "models": summary["models"],
        "baselines": {"market": summary["baselines"]["market"]},
        "comparisons_brier": comparison,
        "base_brier": float(summary["models"]["base"]["brier"]),
        "trained_seed_brier": {
            seed: float(summary["models"][f"seed_{seed}"]["brier"]) for seed in SEEDS
        },
        "trained_seed_mean_brier": trained_mean,
        "market_brier": market_brier,
    }


def finalize(
    registration_path: Path = REGISTRATION,
    execution_path: Path = EXECUTION,
    output_path: Path = OUTPUT,
) -> dict[str, Any]:
    registration, execution = validate_registration(registration_path, execution_path)
    models: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        evaluation_job_id = execution[model_key]["evaluation"]["job_id"]
        summary_path, raw_path = resolve_report(model_key, evaluation_job_id)
        models[model_key] = validate_model(
            model_key,
            registration,
            execution,
            summary_path,
            raw_path,
        )
    market_scores = {models[key]["market_brier"] for key in MODEL_KEYS}
    if len(market_scores) != 1:
        raise RuntimeError("new local models use different market baselines")

    result = {
        "status": "pass",
        "protocol_version": registration["protocol_version"],
        "classification": registration["classification"],
        "registration": str(registration_path),
        "registration_sha256": sha256(registration_path),
        "execution": str(execution_path),
        "execution_sha256": sha256(execution_path),
        "holdout_tasks": str(HOLDOUT),
        "holdout_tasks_sha256": sha256(HOLDOUT),
        "holdout_n": 318,
        "market_brier": market_scores.pop(),
        "models": models,
        "finalizer": str(Path(__file__)),
        "finalizer_sha256": sha256(Path(__file__)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--execution", type=Path, default=EXECUTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = finalize(args.registration, args.execution, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"summary -> {args.output}")


if __name__ == "__main__":
    main()
