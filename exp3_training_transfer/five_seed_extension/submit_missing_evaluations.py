#!/usr/bin/env python3
"""Submit only the evaluation tail after old Slurm dependencies aged out."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("five_seed_launch", HERE / "launch.py")
assert SPEC and SPEC.loader
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)

ACTIVE_PARENT_MODELS = {"qwen3_1_7b", "llama3_2_3b"}


def with_live_dependencies(command: list[str], job_ids: list[str]) -> list[str]:
    result = list(command)
    positions = [index for index, value in enumerate(result) if value.startswith("--dependency=")]
    if len(positions) != 1:
        raise AssertionError("expected one dependency argument")
    result[positions[0]] = "--dependency=afterok:" + ":".join(job_ids)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    path = args.ledger.resolve()
    ledger = json.loads(path.read_text())
    if ledger.get("status") != "submitting" or len(ledger.get("training_jobs", [])) != 84:
        raise SystemExit("evaluation recovery requires the complete training roster")
    if len(ledger.get("probe_jobs", [])) != 48 or ledger.get("exp4_evaluations"):
        raise SystemExit("evaluation recovery requires 48 probes and zero evaluations")

    added = {
        (row["campaign"].removeprefix("exp4_"), int(row["seed"])): str(row["job_id"])
        for row in ledger["training_jobs"]
        if row["campaign"].startswith("exp4_")
    }
    recovery = {
        "recorded_at": launch.now(),
        "cause": "Slurm rejected dependencies on completed parent jobs aged out of the live controller",
        "scientific_change": False,
        "dependency_rule": (
            "completed parent adapters remain in ADAPTER_SPECS; dependencies wait on new seeds only. "
            "Still-running architecture parents remain explicit dependencies."
        ),
        "script_sha256": launch.sha256(Path(__file__).resolve()),
        "submitted_jobs": [],
    }
    ledger["evaluation_recovery"] = recovery
    launch.atomic_json(path, ledger)

    for model_key, item in launch.EXP4_MODELS.items():
        if model_key == "qwen3_32b":
            command = launch.cpu_command(
                HERE / "schedule_qwen32_eval.sbatch",
                {"FIVE_SEED_LEDGER": str(path)},
                f"afterok:{launch.QWEN32_GATE_JOB}",
            )
            job_id = launch.submit(command, True)
            record = {
                "model_key": model_key,
                "kind": "dynamic_scheduler",
                "job_id": job_id,
                "command": command,
            }
        else:
            jobs = {
                **item["parent"],
                **{seed: added[(model_key, seed)] for seed in launch.SEEDS},
            }
            command = launch.evaluation_command(model_key, jobs)
            if model_key in ACTIVE_PARENT_MODELS:
                live = [jobs[seed] for seed in launch.ALL_SEEDS]
            else:
                live = [jobs[seed] for seed in launch.SEEDS]
            command = with_live_dependencies(command, live)
            job_id = launch.submit(command, True)
            record = {
                "model_key": model_key,
                "kind": "evaluation",
                "adapter_jobs": jobs,
                "live_dependency_jobs": live,
                "job_id": job_id,
                "command": command,
            }
        ledger["exp4_evaluations"].append(record)
        recovery["submitted_jobs"].append(
            {"model_key": model_key, "job_id": job_id, "kind": record["kind"]}
        )
        launch.atomic_json(path, ledger)

    ledger["counts"] = {
        "training": 84,
        "probe_extraction": 48,
        "probe_analysis": 48,
        "exp4_evaluation_or_scheduler": 7,
    }
    ledger["status"] = "submitted"
    launch.atomic_json(path, ledger)
    print(f"evaluation tail complete -> {path}")


if __name__ == "__main__":
    main()
