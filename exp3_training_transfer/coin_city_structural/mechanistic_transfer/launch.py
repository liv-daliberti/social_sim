#!/usr/bin/env python3
"""Validate and submit the backfill-sized frozen Qwen3-8B mechanism graph."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    adapter_freeze,
    atomic_json,
    file_sha256,
    load_frozen_tasks,
)

SCRIPTS = {
    "smoke": HERE / "run_smoke.sbatch",
    "extract": HERE / "run_seed.sbatch",
    "behavior": HERE / "run_behavior.sbatch",
    "select": HERE / "run_select.sbatch",
    "patch": HERE / "run_patch.sbatch",
    "finalize": HERE / "run_finalize.sbatch",
}
GPU_TYPES = {"matched": "a6000", "prior": "a40", "base": "a6000"}
SCHEDULER_ROUTE = ("--partition=all", "--account=mltheory")
PATCH_SHARDS = 8


def submit(arguments: list[str], *, enabled: bool) -> tuple[list[str], str | None]:
    command = ["sbatch", "--parsable", *SCHEDULER_ROUTE, *arguments]
    if not enabled:
        return command, None
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return command, completed.stdout.strip().split(";")[0]


def dependency(kind: str, ids: list[str | None], placeholder: str) -> str:
    if all(ids):
        return f"--dependency={kind}:" + ":".join(str(value) for value in ids)
    return f"--dependency={kind}:{placeholder}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    tasks, manifest = load_frozen_tasks()
    protocol_path = HERE / "protocol" / "frozen_protocol.json"
    receipt_path = HERE / "protocol" / "freeze_receipt.json"
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("frozen protocol hash mismatch")
    if protocol["tasks"]["tasks_sha256"] != manifest["tasks_sha256"] or len(tasks) != 512:
        raise ValueError("frozen task mismatch")
    for script in SCRIPTS.values():
        if not script.is_file():
            raise FileNotFoundError(script)
    endpoints = {
        (seed, arm): adapter_freeze(seed, arm)
        for seed in TRAINING_SEEDS
        for arm in ("matched", "prior")
    }

    jobs = []
    smoke_command, smoke_id = submit(
        ["--gres=gpu:a6000:1", str(SCRIPTS["smoke"])], enabled=args.submit
    )
    jobs.append({"stage": "extraction_smoke", "command": smoke_command, "job_id": smoke_id})
    smoke_dependency = dependency("afterok", [smoke_id], "SMOKE_JOB_ID")

    extraction_ids = []
    behavior_ids = []
    for seed in TRAINING_SEEDS:
        for arm in ("matched", "prior"):
            gpu_type = GPU_TYPES[arm]
            endpoint = endpoints[(seed, arm)]
            extract_command, extract_id = submit(
                [
                    smoke_dependency,
                    f"--gres=gpu:{gpu_type}:1",
                    f"--export=ALL,SEED={seed},ARM={arm}",
                    str(SCRIPTS["extract"]),
                ],
                enabled=args.submit,
            )
            extraction_ids.append(extract_id)
            jobs.append(
                {
                    "stage": "activation_extraction",
                    "seed": seed,
                    "arm": arm,
                    "gpu_type": gpu_type,
                    "endpoint": endpoint,
                    "command": extract_command,
                    "job_id": extract_id,
                }
            )
            behavior_command, behavior_id = submit(
                [
                    smoke_dependency,
                    f"--gres=gpu:{gpu_type}:1",
                    f"--export=ALL,SEED={seed},ARM={arm},ADAPTER={endpoint['path']}",
                    str(SCRIPTS["behavior"]),
                ],
                enabled=args.submit,
            )
            behavior_ids.append(behavior_id)
            jobs.append(
                {
                    "stage": "five_draw_behavior",
                    "seed": seed,
                    "arm": arm,
                    "gpu_type": gpu_type,
                    "command": behavior_command,
                    "job_id": behavior_id,
                }
            )

    base_extract_command, base_extract_id = submit(
        [
            smoke_dependency,
            "--gres=gpu:a6000:1",
            "--export=ALL,SEED=42,ARM=base",
            str(SCRIPTS["extract"]),
        ],
        enabled=args.submit,
    )
    base_behavior_command, base_behavior_id = submit(
        [
            smoke_dependency,
            "--gres=gpu:a6000:1",
            "--export=ALL,SEED=42,ARM=base",
            str(SCRIPTS["behavior"]),
        ],
        enabled=args.submit,
    )
    jobs.extend(
        [
            {
                "stage": "base_activation_diagnostic",
                "seed": 42,
                "command": base_extract_command,
                "job_id": base_extract_id,
            },
            {
                "stage": "base_behavior_diagnostic",
                "seed": 42,
                "command": base_behavior_command,
                "job_id": base_behavior_id,
            },
        ]
    )

    select_command, select_id = submit(
        [
            dependency("afterok", extraction_ids, "EXTRACTION_JOB_IDS"),
            str(SCRIPTS["select"]),
        ],
        enabled=args.submit,
    )
    jobs.append({"stage": "select_and_probe_test", "command": select_command, "job_id": select_id})
    patch_smoke_command, patch_smoke_id = submit(
        [
            dependency("afterok", [select_id], "SELECT_JOB_ID"),
            "--gres=gpu:a6000:1",
            "--export=ALL,SEED=42,PATCH_LIMIT=1",
            str(SCRIPTS["patch"]),
        ],
        enabled=args.submit,
    )
    jobs.append(
        {"stage": "patch_smoke", "seed": 42, "command": patch_smoke_command, "job_id": patch_smoke_id}
    )

    patch_array_ids = []
    patch_gpu = {42: "a6000", 43: "a40", 44: "a6000"}
    for seed in TRAINING_SEEDS:
        command, job_id = submit(
            [
                dependency("afterok", [patch_smoke_id], "PATCH_SMOKE_JOB_ID"),
                f"--array=0-{PATCH_SHARDS - 1}",
                f"--gres=gpu:{patch_gpu[seed]}:1",
                f"--export=ALL,SEED={seed},PATCH_SHARDS={PATCH_SHARDS}",
                str(SCRIPTS["patch"]),
            ],
            enabled=args.submit,
        )
        patch_array_ids.append(job_id)
        jobs.append(
            {
                "stage": "causal_patch_array",
                "seed": seed,
                "gpu_type": patch_gpu[seed],
                "shards": PATCH_SHARDS,
                "command": command,
                "job_id": job_id,
            }
        )

    final_prerequisites = [
        *behavior_ids,
        base_extract_id,
        base_behavior_id,
        *patch_array_ids,
    ]
    final_command, final_id = submit(
        [
            dependency("afterok", final_prerequisites, "BEHAVIOR_BASE_PATCH_JOB_IDS"),
            str(SCRIPTS["finalize"]),
        ],
        enabled=args.submit,
    )
    jobs.append({"stage": "final_gate", "command": final_command, "job_id": final_id})

    code_files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sbatch"))
    ledger = {
        "study": STUDY,
        "status": "submitted" if args.submit else "validated_dry_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": file_sha256(protocol_path),
        "tasks_sha256": manifest["tasks_sha256"],
        "scheduler_strategy": (
            "all partition under mltheory account; every GPU task requests <=55 minutes; "
            "trained endpoints and reciprocal donor-pair patches are independently restartable"
        ),
        "paper_external": True,
        "code_sha256": {path.name: file_sha256(path) for path in code_files},
        "jobs": jobs,
    }
    if args.submit:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ledger_path = RUNS_DIR / f"submission_{timestamp}.json"
        atomic_json(ledger_path, ledger)
        ledger["ledger_path"] = str(ledger_path)
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
