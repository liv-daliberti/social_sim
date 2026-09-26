#!/usr/bin/env python3
"""Fail-closed launcher for the Qwen3-14B Exp4 scale extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REGISTERED_DATA = ROOT / "data" / "exp3b_registered"
SCALE_DATA = ROOT / "data" / "exp4_scale_registered"
PROTOCOL = ROOT / "SCALE_PROTOCOL.md"
TRAIN = ROOT / "scripts" / "polymarket_rl_scale_extension.sh"
ADVANCE = ROOT / "scripts" / "advance_scale_extension.sbatch"
LOCKED = ROOT / "scripts" / "locked_test_scale_extension.sbatch"
FINALIZER = ROOT / "scripts" / "finalize_scale_extension.sbatch"
EVALUATOR = ROOT / "scripts" / "evaluate_scale_holdout.py"
RENDERER = ROOT / "scripts" / "render_scale_extension.py"
BUILDER = ROOT / "scripts" / "build_exp4_scale_holdout.py"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"
PROTOCOL_VERSION = "exp4_qwen_scale_v1"
SEEDS = (42, 43, 44)
QWEN14_KEY = "qwen3_14b"
QWEN14_MODEL = "Qwen/Qwen3-14B"
EXISTING_MODELS = {
    "qwen3_4b": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "template": "biased_news",
        "summary": REPORTS / "exp3b_locked_test_j30505540.summary.json",
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "template": "auto_no_think",
        "summary": REPORTS / "exp3b_qwen3_8b_locked_test_j30856250.summary.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(model_id: str) -> Path:
    return HF_HUB / ("models--" + model_id.replace("/", "--"))


def validate_scale_inputs() -> dict[str, Any]:
    manifest_path = SCALE_DATA / "manifest.json"
    task_path = SCALE_DATA / "test.tasks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise SystemExit("scale holdout manifest is not a pass")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("scale holdout protocol drifted")
    if manifest["hashes"]["protocol_sha256"] != sha256(PROTOCOL):
        raise SystemExit("scale protocol changed after holdout construction")
    if manifest["hashes"]["builder_sha256"] != sha256(BUILDER):
        raise SystemExit("scale holdout builder changed after construction")
    if manifest["hashes"]["test_tasks_sha256"] != sha256(task_path):
        raise SystemExit("scale holdout task hash drifted")
    if int(manifest["holdout"]["count"]) < 256:
        raise SystemExit("scale holdout is too small")

    registered_manifest_path = REGISTERED_DATA / "manifest.json"
    if manifest["registered_parent"]["manifest_sha256"] != sha256(
        registered_manifest_path
    ):
        raise SystemExit("registered parent manifest drifted")
    registered_manifest = json.loads(
        registered_manifest_path.read_text(encoding="utf-8")
    )
    for split in ("train", "dev", "test"):
        path = REGISTERED_DATA / f"{split}.tasks.jsonl"
        expected = registered_manifest["output_sha256"][path.name]
        if sha256(path) != expected:
            raise SystemExit(f"registered {split} split drifted")
        if manifest["registered_parent"]["split_sha256"][split] != expected:
            raise SystemExit(f"scale manifest parent {split} hash drifted")
    return manifest


def complete_model_snapshot(snapshot: Path) -> bool:
    if not (snapshot / "config.json").is_file():
        return False
    if not (snapshot / "tokenizer_config.json").is_file():
        return False
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weights = set(index["weight_map"].values())
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
    else:
        weights = {path.name for path in snapshot.glob("*.safetensors")}
    return bool(weights) and all((snapshot / name).is_file() for name in weights)


def validate_model_cache(model_id: str) -> None:
    path = cache_path(model_id)
    snapshots = sorted((path / "snapshots").glob("*")) if path.is_dir() else []
    if not any(complete_model_snapshot(snapshot) for snapshot in snapshots):
        raise SystemExit(f"offline model cache missing or incomplete: {model_id}")


def validate_existing_adapters() -> tuple[dict[str, str], dict[str, Any]]:
    specs: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for model_key, config in EXISTING_MODELS.items():
        validate_model_cache(str(config["model"]))
        summary_path = Path(config["summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("model") != config["model"]:
            raise SystemExit(f"{model_key} frozen summary model drifted")
        adapters = summary.get("adapter_paths", {})
        if sorted(adapters) != ["42", "43", "44"]:
            raise SystemExit(f"{model_key} frozen adapter roster drifted")
        parts = []
        adapter_hashes = {}
        for seed in SEEDS:
            path = Path(adapters[str(seed)])
            weights = path / "adapter_model.safetensors"
            if not (path / "adapter_config.json").is_file() or not weights.is_file():
                raise SystemExit(f"{model_key} seed {seed} adapter is incomplete")
            parts.append(f"{seed}:path={path}")
            adapter_hashes[str(seed)] = sha256(weights)
        specs[model_key] = ";".join(parts)
        audit[model_key] = {
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "adapter_sha256": adapter_hashes,
        }
    return specs, audit


def submit(command: list[str], dry_run: bool) -> str:
    print(shlex.join(command), flush=True)
    if dry_run:
        return "DRYRUN"
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def job_state(job_id: str) -> str:
    output = subprocess.check_output(
        [
            "sacct",
            "-j",
            job_id,
            "--noheader",
            "-X",
            "--format=State",
        ],
        text=True,
    )
    states = [
        line.strip().split()[0].split("+")[0]
        for line in output.splitlines()
        if line.strip()
    ]
    if not states:
        raise SystemExit(f"Slurm has no record of canary job {job_id}")
    return states[0]


def canary_root(job_id: str) -> Path:
    roots = sorted(REPORTS.glob(f"scale_canary_{QWEN14_KEY}_market_*_j{job_id}"))
    if len(roots) != 1:
        raise SystemExit(f"expected one canary report for job {job_id}, got {roots}")
    return roots[0]


def parse_probability(text: str) -> float | None:
    try:
        value = json.loads(text)["yes_prob"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    return (
        probability
        if math.isfinite(probability) and 0.0 <= probability <= 1.0
        else None
    )


def validate_canary(job_id: str) -> dict[str, Any]:
    state = job_state(job_id)
    if state != "COMPLETED":
        raise SystemExit(f"canary job {job_id} is {state}, not COMPLETED")
    root = canary_root(job_id)
    adapters = sorted(root.glob("debug_*/saved_models/step_*"))
    if not adapters or not (adapters[-1] / "adapter_model.safetensors").is_file():
        raise SystemExit("canary did not save a valid final adapter")
    eval_files = sorted(
        root.glob("debug_*/eval_results/*.json"),
        key=lambda path: int(path.stem),
    )
    if not eval_files:
        raise SystemExit("canary did not write a development evaluation")
    rows = json.loads(eval_files[-1].read_text(encoding="utf-8"))
    if len(rows) != 512:
        raise SystemExit("canary development evaluation is incomplete")
    parsed = [
        parse_probability(output) for row in rows for output in row.get("output", [])
    ]
    if len(parsed) != 512 * 5 or any(value is None for value in parsed):
        raise SystemExit("canary did not parse all 2,560 development draws")
    scores = [float(score) for row in rows for score in row.get("scores", [])]
    if len(scores) != 512 * 5 or any(not math.isfinite(score) for score in scores):
        raise SystemExit("canary development scores are incomplete or nonfinite")
    return {
        "job_id": job_id,
        "state": state,
        "report": str(root),
        "final_eval": str(eval_files[-1]),
        "adapter_sha256": sha256(adapters[-1] / "adapter_model.safetensors"),
        "development_parse_coverage": 1.0,
    }


def write_ledger(kind: str, payload: dict[str, Any]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"exp4_scale_{kind}_{timestamp}.json"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "submitted_at": timestamp,
        "protocol_sha256": sha256(PROTOCOL),
        "holdout_manifest_sha256": sha256(SCALE_DATA / "manifest.json"),
        "holdout_tasks_sha256": sha256(SCALE_DATA / "test.tasks.jsonl"),
        "builder_sha256": sha256(BUILDER),
        "train_script_sha256": sha256(TRAIN),
        "advance_script_sha256": sha256(ADVANCE),
        "locked_script_sha256": sha256(LOCKED),
        "finalizer_script_sha256": sha256(FINALIZER),
        "evaluator_sha256": sha256(EVALUATOR),
        "renderer_sha256": sha256(RENDERER),
        "launcher_sha256": sha256(Path(__file__)),
        **payload,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ledger -> {path}")
    return path


def canary_command(args: argparse.Namespace) -> list[str]:
    env = {
        "MODEL_KEY": QWEN14_KEY,
        "MODEL": QWEN14_MODEL,
        "PROMPT_TEMPLATE": "auto_no_think",
        "SEED": "42",
        "RUN_KIND": "canary",
        "MAX_TRAIN": "160",
        "EVAL_STEPS": "10",
        "VLLM_RATIO": str(args.vllm_ratio),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        "--gres=gpu:a6000:2",
        f"--time={args.canary_walltime}",
        f"--exclude={args.exclude}",
        f"--export={export}",
        str(TRAIN),
    ]


def launch_canary(args: argparse.Namespace) -> None:
    validate_scale_inputs()
    validate_model_cache(QWEN14_MODEL)
    command = canary_command(args)
    job_id = submit(command, args.dry_run)
    write_ledger(
        "canary",
        {
            "dry_run": args.dry_run,
            "job_id": job_id,
            "command": command,
            "training_hyperparameters_changed_from_8b": False,
            "development_monitor": "five_draw_stochastic",
        },
    )
    print(f"submitted scale canary {job_id}")


def canary_ledger(job_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(RUNS.glob("exp4_scale_canary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("job_id")) == job_id:
            matches.append((path, payload))
    if len(matches) != 1:
        raise SystemExit(
            f"expected one registered canary ledger for {job_id}, got {len(matches)}"
        )
    path, payload = matches[0]
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256(PROTOCOL),
        "holdout_manifest_sha256": sha256(SCALE_DATA / "manifest.json"),
        "holdout_tasks_sha256": sha256(SCALE_DATA / "test.tasks.jsonl"),
        "builder_sha256": sha256(BUILDER),
        "train_script_sha256": sha256(TRAIN),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"canary ledger {key} drifted: {path}")
    if payload.get("dry_run") is not False:
        raise SystemExit(f"canary ledger is not a real submission: {path}")
    return path, payload


def launch_advance(args: argparse.Namespace) -> None:
    validate_scale_inputs()
    validate_model_cache(QWEN14_MODEL)
    canary_path, _ = canary_ledger(args.canary_job_id)
    env = {
        "CANARY_JOB_ID": args.canary_job_id,
        "SCALE_PARTITION": args.partition,
        "SCALE_EXCLUDE": args.exclude,
        "SCALE_VLLM_RATIO": str(args.vllm_ratio),
        "SCALE_TRAIN_WALLTIME": args.walltime,
        "SCALE_EVAL_WALLTIME": args.eval_walltime,
        "SCALE_FINALIZE_WALLTIME": args.finalize_walltime,
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--time={args.advance_walltime}",
        "--mem=4G",
        "--cpus-per-task=1",
        f"--exclude={args.exclude}",
        f"--dependency=afterok:{args.canary_job_id}",
        f"--export={export}",
        str(ADVANCE),
    ]
    job_id = submit(command, args.dry_run)
    write_ledger(
        "advance",
        {
            "dry_run": args.dry_run,
            "job_id": job_id,
            "canary_job_id": args.canary_job_id,
            "canary_ledger": str(canary_path),
            "environment": env,
            "command": command,
        },
    )
    print(f"submitted canary-gated scale advance {job_id}")


def launch_full(args: argparse.Namespace) -> None:
    holdout_manifest = validate_scale_inputs()
    validate_model_cache(QWEN14_MODEL)
    existing_specs, existing_audit = validate_existing_adapters()
    if args.dry_run:
        canary_audit = {"job_id": args.canary_job_id, "state": "DRYRUN"}
    else:
        canary_audit = validate_canary(args.canary_job_id)

    training_jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        env = {
            "MODEL_KEY": QWEN14_KEY,
            "MODEL": QWEN14_MODEL,
            "PROMPT_TEMPLATE": "auto_no_think",
            "SEED": str(seed),
            "RUN_KIND": "train",
            "MAX_TRAIN": "4800",
            "EVAL_STEPS": "25",
            "VLLM_RATIO": str(args.vllm_ratio),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partition}",
            "--gres=gpu:a6000:2",
            f"--time={args.walltime}",
            f"--exclude={args.exclude}",
            f"--export={export}",
            str(TRAIN),
        ]
        job_id = submit(command, args.dry_run)
        training_jobs.append(
            {
                "seed": seed,
                "job_id": job_id,
                "environment": env,
                "command": command,
            }
        )

    dependencies = ":".join(
        item["job_id"] for item in training_jobs if item["job_id"] != "DRYRUN"
    )
    qwen14_specs = ";".join(
        f"{item['seed']}:job={item['job_id']}" for item in training_jobs
    )
    roster = {
        **{
            key: {
                "model": str(config["model"]),
                "template": str(config["template"]),
                "adapter_specs": existing_specs[key],
            }
            for key, config in EXISTING_MODELS.items()
        },
        QWEN14_KEY: {
            "model": QWEN14_MODEL,
            "template": "auto_no_think",
            "adapter_specs": qwen14_specs,
        },
    }
    evaluation_jobs: list[dict[str, Any]] = []
    for model_key, config in roster.items():
        env = {
            "MODEL_KEY": model_key,
            "MODEL": config["model"],
            "PROMPT_TEMPLATE": config["template"],
            "ADAPTER_SPECS": config["adapter_specs"],
        }
        export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partition}",
            "--gres=gpu:a6000:1",
            f"--time={args.eval_walltime}",
            f"--exclude={args.exclude}",
        ]
        if dependencies:
            command.append(f"--dependency=afterok:{dependencies}")
        command.extend([f"--export={export}", str(LOCKED)])
        job_id = submit(command, args.dry_run)
        evaluation_jobs.append(
            {
                "model_key": model_key,
                "job_id": job_id,
                "environment": env,
                "command": command,
            }
        )

    evaluation_dependencies = ":".join(
        item["job_id"] for item in evaluation_jobs if item["job_id"] != "DRYRUN"
    )
    evaluation_specs = ";".join(
        f"{item['model_key']}:{item['job_id']}" for item in evaluation_jobs
    )
    finalizer_env = {"EVALUATION_SPECS": evaluation_specs}
    finalizer_export = "ALL," + ",".join(
        f"{key}={value}" for key, value in finalizer_env.items()
    )
    finalizer_command = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--time={args.finalize_walltime}",
        "--mem=16G",
        "--cpus-per-task=2",
        f"--exclude={args.exclude}",
    ]
    if evaluation_dependencies:
        finalizer_command.append(f"--dependency=afterok:{evaluation_dependencies}")
    finalizer_command.extend([f"--export={finalizer_export}", str(FINALIZER)])
    finalizer_job_id = submit(finalizer_command, args.dry_run)
    finalization_job = {
        "job_id": finalizer_job_id,
        "environment": finalizer_env,
        "command": finalizer_command,
    }

    write_ledger(
        "full",
        {
            "dry_run": args.dry_run,
            "holdout_public_description": holdout_manifest["holdout"],
            "canary": canary_audit,
            "existing_adapters": existing_audit,
            "training_jobs": training_jobs,
            "evaluation_jobs": evaluation_jobs,
            "finalization_job": finalization_job,
        },
    )
    print(
        f"submitted {len(training_jobs)} training and "
        f"{len(evaluation_jobs)} dependency-locked evaluations; "
        f"finalizer {finalizer_job_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("canary", "advance", "full"), required=True)
    parser.add_argument("--canary-job-id", default="")
    parser.add_argument("--partition", default="all")
    parser.add_argument("--exclude", default="node206")
    parser.add_argument("--canary-walltime", default="06:00:00")
    parser.add_argument("--walltime", default="30:00:00")
    parser.add_argument("--eval-walltime", default="08:00:00")
    parser.add_argument("--finalize-walltime", default="01:00:00")
    parser.add_argument("--advance-walltime", default="00:30:00")
    parser.add_argument("--vllm-ratio", type=float, default=0.78)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.70 <= args.vllm_ratio <= 0.85:
        parser.error("--vllm-ratio must be in [0.70, 0.85]")
    if args.stage in {"advance", "full"} and not args.canary_job_id:
        parser.error(f"--stage {args.stage} requires --canary-job-id")
    if args.stage == "canary":
        launch_canary(args)
    elif args.stage == "advance":
        launch_advance(args)
    else:
        launch_full(args)


if __name__ == "__main__":
    main()
