#!/usr/bin/env python3
"""Canary-gated launcher for the mechanism-family large-model scale extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_canaries


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
REPO = FAMILY.parents[1]
PROBE = FAMILY / "mechanistic_probe"
REPORTS = FAMILY / "reports"
RUNS = HERE / "runs"
LOGS = HERE / "logs"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"
PROTOCOL = "c3_mechanism_scale_v2_1"
DISCLOSURES = ("disclosed", "undisclosed")
ARMS = ("causal_family", "population_prior")
SEEDS = (42, 43, 44)


MODELS: dict[str, dict[str, Any]] = {
    "qwen3_14b": {
        "id": "Qwen/Qwen3-14B",
        "label": "Qwen3-14B",
        "commit": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "template": "auto_no_think",
        "train": {"gpus": 2, "cpus": 12, "memory": "160G", "canary": "06:00:00", "full": "36:00:00"},
        "base": {"gpus": 1, "cpus": 12, "memory": "64G", "time": "06:00:00"},
        "extract": {"gpus": 1, "cpus": 12, "memory": "80G", "time": "12:00:00", "device_map": "none"},
    },
    "qwen3_32b": {
        "id": "Qwen/Qwen3-32B",
        "label": "Qwen3-32B",
        "commit": "9216db5781bf21249d130ec9da846c4624c16137",
        "template": "auto_no_think",
        "train": {"gpus": 4, "cpus": 16, "memory": "260G", "canary": "08:00:00", "full": "48:00:00"},
        "base": {"gpus": 2, "cpus": 16, "memory": "140G", "time": "08:00:00"},
        "extract": {"gpus": 2, "cpus": 16, "memory": "180G", "time": "18:00:00", "device_map": "balanced"},
    },
    "llama3_1_70b": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "label": "Llama-3.1-70B-Instruct",
        "commit": "1605565b47bb9346c5515c34102e054115b4f98b",
        "template": "auto",
        "train": {"gpus": 8, "cpus": 24, "memory": "460G", "canary": "16:00:00", "full": "72:00:00"},
        "base": {"gpus": 4, "cpus": 24, "memory": "260G", "time": "16:00:00"},
        "extract": {"gpus": 4, "cpus": 24, "memory": "320G", "time": "30:00:00", "device_map": "balanced"},
    },
}


HASH_PATHS = {
    "protocol": HERE / "PROTOCOL.md",
    "launcher": Path(__file__),
    "trainer_wrapper": HERE / "large_mechanism_rl.sh",
    "base_wrapper": HERE / "large_base_eval.sh",
    "tp_evaluator": HERE / "evaluate_endpoint_tp.py",
    "canary_audit": HERE / "audit_canaries.py",
    "advance": HERE / "advance.sbatch",
    "finalize": HERE / "finalize.sbatch",
    "probe_extract_runner": HERE / "extract_probe.sbatch",
    "probe_analysis_runner": HERE / "analyze_probe.sbatch",
    "output_runner": HERE / "output_tracking.sbatch",
    "mechanism_trainer": FAMILY / "run_mechanism_rl.py",
    "scorer": FAMILY / "report.py",
    "world_catalog": FAMILY / "worlds.py",
    "prompt_renderer": FAMILY / "prompt.py",
    "output_contract": FAMILY / "output_contract.py",
    "data_build_manifest": FAMILY / "data" / "build_manifest.json",
    "mechanism_manifest": FAMILY / "protocol" / "c3_mechanism_manifest.json",
    "probe_tasks": PROBE / "data" / "tasks.jsonl",
    "probe_auxiliary": PROBE / "data" / "auxiliary_targets_v2.jsonl",
    "probe_extractor": PROBE / "extract_hidden_states.py",
    "probe_analysis": PROBE / "analyze_probe_v2.py",
    "output_analysis": PROBE / "analyze_output_tracking.py",
    "output_parse_runner": PROBE / "run_output_tracking_v2.py",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in HASH_PATHS.values() if not path.is_file()]
    if missing:
        raise SystemExit("scale v2.1 implementation incomplete: " + ", ".join(missing))
    return {name: sha256(path) for name, path in HASH_PATHS.items()}


def snapshot_path(model_id: str, commit: str) -> Path:
    cache = HF_HUB / ("models--" + model_id.replace("/", "--"))
    snapshot = cache / "snapshots" / commit
    if not snapshot.is_dir():
        raise SystemExit(f"missing pinned model snapshot: {snapshot}")
    ref = cache / "refs" / "main"
    if not ref.is_file() or ref.read_text().strip() != commit:
        raise SystemExit(f"model ref does not match pinned commit: {model_id}")
    return snapshot


def preflight() -> dict[str, Any]:
    hashes = code_hashes()
    build = json.loads((FAMILY / "data" / "build_manifest.json").read_text())
    if (build.get("n_train_per_world"), build.get("n_eval_per_world")) != (600, 60):
        raise SystemExit("large extension requires the registered 600/60 datasets")
    for disclosure in DISCLOSURES:
        for arm in ARMS:
            root = FAMILY / "data" / f"{disclosure}_{arm}"
            for split in ("train", "heldout"):
                if not (root / split).is_dir():
                    raise SystemExit(f"missing registered dataset: {root / split}")
    snapshots = {
        key: str(snapshot_path(item["id"], item["commit"]))
        for key, item in MODELS.items()
    }
    return {"code_and_data_sha256": hashes, "model_snapshots": snapshots}


def export_arg(environment: dict[str, Any]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in environment.items())


def gpu_command(
    *, script: Path, environment: dict[str, Any], gpus: int, cpus: int,
    memory: str, walltime: str, dependency: str | None = None,
) -> list[str]:
    command = [
        "sbatch", "--parsable", "--account=allcs", "--partition=cs",
        "--qos=medium", f"--gres=gpu:a6000:{gpus}",
        f"--cpus-per-task={cpus}", f"--mem={memory}", f"--time={walltime}",
        "--exclude=node206",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend([f"--export={export_arg(environment)}", str(script)])
    return command


def cpu_command(
    *, script: Path, environment: dict[str, Any], dependency: str | None = None,
) -> list[str]:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend([f"--export={export_arg(environment)}", str(script)])
    return command


def submit(command: list[str], do_submit: bool) -> str | None:
    print(shlex.join(command))
    if not do_submit:
        return None
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {completed.stdout!r}")
    print(f"submitted {job_id}")
    return job_id


def training_environment(
    model_key: str, disclosure: str, arm: str, seed: int, run_kind: str,
) -> dict[str, Any]:
    item = MODELS[model_key]
    return {
        "DISCLOSURE": disclosure,
        "ARM": arm,
        "MODEL_KEY": model_key,
        "MODEL": item["id"],
        "PROMPT_TEMPLATE": item["template"],
        "SEED": seed,
        "RUN_KIND": run_kind,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def training_command(
    model_key: str, disclosure: str, arm: str, seed: int, run_kind: str,
) -> list[str]:
    item = MODELS[model_key]
    resources = item["train"]
    return gpu_command(
        script=HERE / "large_mechanism_rl.sh",
        environment=training_environment(model_key, disclosure, arm, seed, run_kind),
        gpus=resources["gpus"], cpus=resources["cpus"], memory=resources["memory"],
        walltime=resources["canary" if run_kind == "canary" else "full"],
    )


def base_command(model_key: str, disclosure: str) -> list[str]:
    item = MODELS[model_key]
    resources = item["base"]
    environment = {
        "DISCLOSURE": disclosure,
        "MODEL_KEY": model_key,
        "MODEL": item["id"],
        "PROMPT_TEMPLATE": item["template"],
    }
    return gpu_command(
        script=HERE / "large_base_eval.sh", environment=environment,
        gpus=resources["gpus"], cpus=resources["cpus"], memory=resources["memory"],
        walltime=resources["time"],
    )


def scientific_specs() -> list[tuple[str, str, str, int]]:
    return [
        (model, disclosure, arm, seed)
        for model in MODELS
        for disclosure in DISCLOSURES
        for arm in ARMS
        for seed in SEEDS
    ]


def assert_hashes(expected: dict[str, str]) -> None:
    current = code_hashes()
    if current != expected:
        changed = sorted(set(current) | set(expected))
        drift = {key: {"expected": expected.get(key), "observed": current.get(key)}
                 for key in changed if expected.get(key) != current.get(key)}
        raise SystemExit(f"scale v2.1 protocol/code drift: {json.dumps(drift, sort_keys=True)}")


def refuse_existing(pattern: str, label: str) -> None:
    existing = sorted(RUNS.glob(pattern))
    if existing:
        raise SystemExit(f"{label} already exists; refusing duplicate launch: {existing}")


def canary_stage(do_submit: bool) -> None:
    manifest = preflight()
    if do_submit:
        refuse_existing("scale_v2_1_canary_*.json", "canary ledger")
        existing_reports = sorted(REPORTS.glob("scale_v2_1_*"))
        if existing_reports:
            raise SystemExit(f"large-model report state already exists: {existing_reports}")
    records = []
    for model_key in MODELS:
        for disclosure in DISCLOSURES:
            command = training_command(
                model_key, disclosure, "causal_family", 42, "canary"
            )
            job_id = submit(command, do_submit)
            records.append(
                {
                    "kind": "canary", "model_key": model_key,
                    "disclosure": disclosure, "arm": "causal_family", "seed": 42,
                    "environment": training_environment(
                        model_key, disclosure, "causal_family", 42, "canary"
                    ),
                    "command": command, "job_id": job_id,
                }
            )
    for model_key in MODELS:
        for disclosure in DISCLOSURES:
            command = base_command(model_key, disclosure)
            job_id = submit(command, do_submit)
            records.append(
                {
                    "kind": "base_evaluation", "model_key": model_key,
                    "disclosure": disclosure, "command": command, "job_id": job_id,
                }
            )
    if not do_submit:
        print(json.dumps({"stage": "canary", "jobs": len(records), **manifest}, indent=2))
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"scale_v2_1_canary_{timestamp()}.json"
    ledger = {
        "protocol": PROTOCOL, "stage": "canary", "created_at": timestamp(),
        **manifest, "submissions": records,
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ids = [str(row["job_id"]) for row in records]
    advance = cpu_command(
        script=HERE / "advance.sbatch",
        environment={"CANARY_LEDGER": str(path)},
        dependency="afterok:" + ":".join(ids),
    )
    advance_id = submit(advance, True)
    ledger["advance"] = {"job_id": advance_id, "command": advance}
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ledger": str(path), "canary_and_base_jobs": ids,
                      "advance_job_id": advance_id}, indent=2))


def render_full() -> None:
    manifest = preflight()
    commands = [training_command(*spec, "train") for spec in scientific_specs()]
    for command in commands:
        print(shlex.join(command))
    print(json.dumps({"stage": "full-render", "jobs": len(commands), **manifest}, indent=2))


def advance_stage(canary_ledger: Path, do_submit: bool) -> None:
    if not do_submit:
        raise SystemExit("advance is an audited submission stage; use full-render for dry run")
    parent = json.loads(canary_ledger.read_text(encoding="utf-8"))
    if parent.get("protocol") != PROTOCOL:
        raise SystemExit("wrong canary protocol")
    assert_hashes(parent["code_and_data_sha256"])
    refuse_existing("scale_v2_1_full_*.json", "full ledger")
    audit_path = RUNS / f"scale_v2_1_canary_audit_{timestamp()}.json"
    audit = audit_canaries.audit(canary_ledger)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    records = []
    for model_key, disclosure, arm, seed in scientific_specs():
        command = training_command(model_key, disclosure, arm, seed, "train")
        job_id = submit(command, True)
        records.append(
            {
                "kind": "training", "model_key": model_key,
                "disclosure": disclosure, "arm": arm, "seed": seed,
                "environment": training_environment(
                    model_key, disclosure, arm, seed, "train"
                ),
                "command": command, "job_id": job_id,
            }
        )
    path = RUNS / f"scale_v2_1_full_{timestamp()}.json"
    ledger = {
        "protocol": PROTOCOL, "stage": "full", "created_at": timestamp(),
        "parent_canary_ledger": str(canary_ledger),
        "canary_audit": str(audit_path),
        "code_and_data_sha256": parent["code_and_data_sha256"],
        "model_snapshots": parent["model_snapshots"],
        "base_evaluations": [
            row for row in parent["submissions"] if row["kind"] == "base_evaluation"
        ],
        "submissions": records,
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    full_ids = [str(row["job_id"]) for row in records]
    finalizer = cpu_command(
        script=HERE / "finalize.sbatch", environment={"FULL_LEDGER": str(path)},
        dependency="afterok:" + ":".join(full_ids),
    )
    finalizer_id = submit(finalizer, True)
    ledger["finalizer"] = {"job_id": finalizer_id, "command": finalizer}
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ledger": str(path), "full_jobs": full_ids,
                      "finalizer_job_id": finalizer_id}, indent=2))


def completed_training_reports(full: dict[str, Any]) -> list[dict[str, Any]]:
    rows = full["submissions"]
    job_ids = [str(row["job_id"]) for row in rows]
    states = audit_canaries.slurm_states(job_ids)
    incomplete = {job: states.get(job, "UNKNOWN") for job in job_ids
                  if states.get(job) != "COMPLETED"}
    if incomplete:
        raise SystemExit(f"full training incomplete: {incomplete}")
    resolved = []
    for row in rows:
        job_id = str(row["job_id"])
        roots = sorted(REPORTS.glob(f"scale_v2_1_train_*_j{job_id}"))
        if len(roots) != 1:
            raise SystemExit(f"job {job_id}: expected one report, got {roots}")
        adapters = sorted(roots[0].glob("debug_*/saved_models/step_*"))
        if not adapters:
            raise SystemExit(f"job {job_id}: adapter missing")
        scores = roots[0] / "stochastic_n5.scores.jsonl"
        if not scores.is_file():
            raise SystemExit(f"job {job_id}: stochastic scores missing")
        resolved.append({**row, "report": str(roots[0]), "adapter": str(adapters[-1]),
                         "score_file": str(scores)})
    return resolved


def completed_base_reports(full: dict[str, Any]) -> list[dict[str, Any]]:
    rows = full["base_evaluations"]
    job_ids = [str(row["job_id"]) for row in rows]
    states = audit_canaries.slurm_states(job_ids)
    incomplete = {job: states.get(job, "UNKNOWN") for job in job_ids
                  if states.get(job) != "COMPLETED"}
    if incomplete:
        raise SystemExit(f"base evaluations incomplete: {incomplete}")
    resolved = []
    for row in rows:
        job_id = str(row["job_id"])
        root = REPORTS / f"scale_v2_1_base_{row['disclosure']}_{row['model_key']}_j{job_id}"
        scores = root / "stochastic_n5.scores.jsonl"
        if not scores.is_file():
            raise SystemExit(f"job {job_id}: base stochastic scores missing")
        resolved.append({**row, "report": str(root), "score_file": str(scores)})
    return resolved


def extraction_command(
    model_key: str, endpoint: str, run_dir: Path, adapter: str | None,
) -> list[str]:
    item = MODELS[model_key]
    resources = item["extract"]
    environment: dict[str, Any] = {
        "MODEL": item["id"], "MODEL_COMMIT": item["commit"],
        "MODEL_LABEL": item["label"], "ENDPOINT_LABEL": endpoint,
        "RUN_DIR": str(run_dir), "DEVICE_MAP": resources["device_map"],
    }
    if adapter:
        environment["ADAPTER"] = adapter
    return gpu_command(
        script=HERE / "extract_probe.sbatch", environment=environment,
        gpus=resources["gpus"], cpus=resources["cpus"], memory=resources["memory"],
        walltime=resources["time"],
    )


def probe_stage(full_ledger: Path, do_submit: bool) -> None:
    if not do_submit:
        raise SystemExit("probe is an audited submission stage")
    full = json.loads(full_ledger.read_text(encoding="utf-8"))
    if full.get("protocol") != PROTOCOL or full.get("stage") != "full":
        raise SystemExit("wrong full ledger")
    assert_hashes(full["code_and_data_sha256"])
    refuse_existing("scale_v2_1_probe_*.json", "probe ledger")
    training = completed_training_reports(full)
    bases = completed_base_reports(full)
    if len(training) != 36 or len(bases) != 6:
        raise SystemExit("scientific large-model roster is incomplete")

    score_manifest = RUNS / f"scale_v2_1_score_manifest_{timestamp()}.json"
    score_manifest.write_text(
        json.dumps(
            {"protocol": PROTOCOL, "score_files":
             [row["score_file"] for row in training + bases]},
            indent=2, sort_keys=True,
        ) + "\n"
    )

    submissions = []
    for model_key in MODELS:
        endpoints = [("base", None)] + [
            (f"{row['disclosure']}_{row['arm']}_s{row['seed']}", row["adapter"])
            for row in training if row["model_key"] == model_key
        ]
        if len(endpoints) != 13:
            raise SystemExit(f"{model_key}: expected 13 probe endpoints")
        for endpoint, adapter in endpoints:
            run_dir = PROBE / "runs" / f"{model_key}_{endpoint}"
            extract = extraction_command(model_key, endpoint, run_dir, adapter)
            extract_id = submit(extract, True)
            analysis = cpu_command(
                script=HERE / "analyze_probe.sbatch",
                environment={"RUN_DIR": str(run_dir)},
                dependency=f"afterok:{extract_id}",
            )
            analysis_id = submit(analysis, True)
            submissions.append(
                {
                    "model_key": model_key, "endpoint": endpoint,
                    "adapter": adapter, "run_dir": str(run_dir),
                    "extract_job_id": extract_id, "extract_command": extract,
                    "analysis_job_id": analysis_id, "analysis_command": analysis,
                }
            )

    output_jobs = []
    for model_key in MODELS:
        command = cpu_command(
            script=HERE / "output_tracking.sbatch",
            environment={"MODEL_KEY": model_key, "SCORE_MANIFEST": str(score_manifest)},
        )
        job_id = submit(command, True)
        output_jobs.append({"model_key": model_key, "job_id": job_id, "command": command})

    path = RUNS / f"scale_v2_1_probe_{timestamp()}.json"
    ledger = {
        "protocol": PROTOCOL, "stage": "probe", "created_at": timestamp(),
        "parent_full_ledger": str(full_ledger),
        "code_and_data_sha256": full["code_and_data_sha256"],
        "score_manifest": str(score_manifest),
        "submissions": submissions, "output_tracking": output_jobs,
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ledger": str(path), "probe_endpoints": len(submissions),
                      "output_jobs": output_jobs}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("canary", "full-render", "advance", "probe"),
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--canary-ledger", type=Path)
    parser.add_argument("--full-ledger", type=Path)
    args = parser.parse_args()
    if args.stage == "canary":
        canary_stage(args.submit)
    elif args.stage == "full-render":
        if args.submit:
            parser.error("full-render never submits; use the automatic advance gate")
        render_full()
    elif args.stage == "advance":
        if args.canary_ledger is None:
            parser.error("advance requires --canary-ledger")
        advance_stage(args.canary_ledger, args.submit)
    else:
        if args.full_ledger is None:
            parser.error("probe requires --full-ledger")
        probe_stage(args.full_ledger, args.submit)


if __name__ == "__main__":
    main()
