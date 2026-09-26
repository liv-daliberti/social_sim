#!/usr/bin/env python3
"""Resume the single incrementally recorded seed-extension submission."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("five_seed_launch", HERE / "launch.py")
assert SPEC and SPEC.loader
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)

ARCH_COMMITS = {
    "qwen3_1_7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "llama3_2_3b": "006f5dcd1393c3add266de40994ba96225e9689d",
}


def architecture_recovery(row: dict[str, Any]) -> dict[str, Any]:
    model_key = row["identity"]
    if model_key not in ARCH_COMMITS:
        return row
    item = launch.EXP4_MODELS[model_key]
    snapshot = (
        launch.REPO / ".runtime" / "hf_home" / "hub"
        / ("models--" + item["model"].replace("/", "--"))
        / "snapshots" / ARCH_COMMITS[model_key]
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    env = {
        "MODEL_KEY": model_key,
        "MODEL": str(snapshot),
        "MODEL_CHECKPOINT": item["model"],
        "PROMPT_TEMPLATE": item["template"],
        "SEED": str(row["seed"]),
        "RUN_KIND": "train",
        "MAX_TRAIN": "4800",
        "EVAL_STEPS": "25",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    walltime = "10:00:00" if model_key == "llama3_2_3b" else "08:00:00"
    command = launch.gpu_command(
        launch.POLY / "scripts/polymarket_rl_architecture_extension.sh",
        env,
        gpus=1,
        cpus=8,
        memory="72G",
        walltime=walltime,
        partition="cs",
        account="allcs",
        qos="medium",
    )
    return {**row, "environment": env, "command": command}


def finish_graph(ledger: dict[str, Any], path: Path) -> None:
    if ledger.get("probe_jobs") or ledger.get("exp4_evaluations"):
        raise SystemExit("partial ledger already contains downstream jobs")
    for row in ledger["training_jobs"]:
        if row["campaign"] not in ("exp3_mechanism_current", "exp3_mechanism_large"):
            continue
        extract_command, metadata = launch.probe_commands(row)
        extract_id = launch.submit(extract_command, True)
        analysis_command = launch.cpu_command(
            HERE / "analyze_probe.sbatch",
            {"RUN_DIR": metadata["run_dir"]},
            f"afterok:{extract_id}",
        )
        analysis_id = launch.submit(analysis_command, True)
        ledger["probe_jobs"].append(
            {
                **metadata,
                "campaign": row["campaign"],
                "seed": row["seed"],
                "training_job_id": row["job_id"],
                "extract_job_id": extract_id,
                "extract_command": extract_command,
                "analysis_job_id": analysis_id,
                "analysis_command": analysis_command,
            }
        )
        launch.atomic_json(path, ledger)

    added = {
        (row["campaign"].removeprefix("exp4_"), int(row["seed"])): row["job_id"]
        for row in ledger["training_jobs"]
        if row["campaign"].startswith("exp4_")
    }
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
            job_id = launch.submit(command, True)
            record = {
                "model_key": model_key,
                "kind": "evaluation",
                "adapter_jobs": jobs,
                "job_id": job_id,
                "command": command,
            }
        ledger["exp4_evaluations"].append(record)
        launch.atomic_json(path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    path = args.ledger.resolve()
    ledger = json.loads(path.read_text())
    if ledger.get("protocol") != "exp3_exp4_five_seed_extension_v1":
        raise SystemExit("wrong protocol")
    if ledger.get("status") != "submitting" or ledger.get("submitted") is not True:
        raise SystemExit("not a recoverable partial submission")
    if len(ledger.get("training_jobs", [])) != 78:
        raise SystemExit("recovery is locked to the recorded 78/84 partial state")

    planned = launch.training_records()
    existing = {
        (row["campaign"], row["identity"], int(row["seed"]))
        for row in ledger["training_jobs"]
    }
    missing = [
        architecture_recovery(row)
        for row in planned
        if (row["campaign"], row["identity"], int(row["seed"])) not in existing
    ]
    if len(missing) != 6:
        raise SystemExit(f"expected six missing jobs, found {len(missing)}")
    recovery = {
        "resumed_at": launch.now(),
        "prior_training_count": 78,
        "missing_training_count": 6,
        "cause": "mltheory/all rejected the first Qwen3-1.7B allocation",
        "allocation_amendment": (
            "Qwen3-1.7B and Llama-3.2-3B moved to allcs/cs; "
            "pinned snapshots and scientific settings unchanged"
        ),
        "recovery_sha256": launch.sha256(Path(__file__).resolve()),
        "submitted_jobs": [],
    }
    ledger["recovery"] = recovery
    launch.atomic_json(path, ledger)
    for row in missing:
        job_id = launch.submit(row["command"], True)
        ledger["training_jobs"].append({**row, "job_id": job_id})
        recovery["submitted_jobs"].append(
            {"campaign": row["campaign"], "identity": row["identity"],
             "seed": row["seed"], "job_id": job_id}
        )
        launch.atomic_json(path, ledger)

    if len(ledger["training_jobs"]) != 84:
        raise AssertionError("training recovery did not reach 84 jobs")
    finish_graph(ledger, path)
    ledger["counts"] = {
        "training": 84,
        "probe_extraction": len(ledger["probe_jobs"]),
        "probe_analysis": len(ledger["probe_jobs"]),
        "exp4_evaluation_or_scheduler": len(ledger["exp4_evaluations"]),
    }
    ledger["status"] = "submitted"
    launch.atomic_json(path, ledger)
    print(f"recovered complete graph -> {path}")


if __name__ == "__main__":
    main()
