#!/usr/bin/env python3
"""Fail-closed launcher for the prospective dense Qwen3-32B extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "exp4_scale_registered"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"
PROTOCOL_VERSION = "exp4_qwen32_scale_v1"
PARENT_PROTOCOL_VERSION = "exp4_qwen_scale_v1"
PROTOCOL = ROOT / "QWEN32_SCALE_PROTOCOL.md"
PARENT_PROTOCOL = ROOT / "SCALE_PROTOCOL.md"
PARENT_LEDGER = RUNS / "exp4_scale_full_20260824T211347Z.json"
TRAIN = ROOT / "scripts" / "polymarket_rl_qwen32_extension.sh"
SHARED_TRAINER = (
    REPO / "exp3_training_transfer" / "mechanism_family" / "run_mechanism_rl.py"
)
OAT_DEEPSPEED = REPO / ".runtime" / "oat" / "oat" / "utils" / "deepspeed.py"
ADVANCE = ROOT / "scripts" / "advance_qwen32_extension.sbatch"
EVALUATOR_WRAPPER = ROOT / "scripts" / "evaluate_qwen32_holdout.py"
FROZEN_EVALUATOR = ROOT / "scripts" / "evaluate_scale_holdout.py"
LOCKED = ROOT / "scripts" / "locked_test_qwen32_extension.sbatch"
RELEASE = ROOT / "scripts" / "release_qwen32_locked_evals.sbatch"
RENDERER = ROOT / "scripts" / "render_qwen32_extension.py"
FINALIZER = ROOT / "scripts" / "finalize_qwen32_extension.sbatch"
MODEL_KEY = "qwen3_32b"
MODEL = "Qwen/Qwen3-32B"
SEEDS = (42, 43, 44)
PARENT_EVALUATIONS = {
    "qwen3_4b": "30870546",
    "qwen3_8b": "30870547",
    "qwen3_14b": "30870548",
}
PARENT_FINALIZER = "30870549"
HELD_JOBS = (*PARENT_EVALUATIONS.values(), PARENT_FINALIZER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "parent_ledger": PARENT_LEDGER,
        "trainer": TRAIN,
        "shared_trainer": SHARED_TRAINER,
        "oat_deepspeed": OAT_DEEPSPEED,
        "advance": ADVANCE,
        "evaluator_wrapper": EVALUATOR_WRAPPER,
        "frozen_parent_evaluator": FROZEN_EVALUATOR,
        "locked_evaluator": LOCKED,
        "release": RELEASE,
        "renderer": RENDERER,
        "finalizer": FINALIZER,
        "launcher": Path(__file__),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Qwen3-32B implementation is incomplete: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def cache_path(model_id: str) -> Path:
    return HF_HUB / ("models--" + model_id.replace("/", "--"))


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


def validate_model_cache() -> dict[str, Any]:
    path = cache_path(MODEL)
    snapshots = sorted((path / "snapshots").glob("*")) if path.is_dir() else []
    complete = [snapshot for snapshot in snapshots if complete_model_snapshot(snapshot)]
    if len(complete) != 1:
        raise SystemExit(
            f"expected one complete offline {MODEL} snapshot, got {complete}"
        )
    return {
        "model": MODEL,
        "cache_path": str(path),
        "snapshot": str(complete[0]),
        "revision": complete[0].name,
        "config_sha256": sha256(complete[0] / "config.json"),
    }


def validate_parent_holdout() -> dict[str, Any]:
    manifest_path = DATA / "manifest.json"
    task_path = DATA / "test.tasks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise SystemExit("parent scale holdout manifest is not a pass")
    if manifest.get("protocol_version") != PARENT_PROTOCOL_VERSION:
        raise SystemExit("parent scale holdout protocol drifted")
    if manifest["hashes"]["protocol_sha256"] != sha256(PARENT_PROTOCOL):
        raise SystemExit("parent scale protocol changed after holdout construction")
    if manifest["hashes"]["test_tasks_sha256"] != sha256(task_path):
        raise SystemExit("parent scale holdout task hash drifted")
    if int(manifest["holdout"]["count"]) < 256:
        raise SystemExit("parent scale holdout is too small")

    parent = json.loads(PARENT_LEDGER.read_text(encoding="utf-8"))
    if parent.get("protocol_version") != PARENT_PROTOCOL_VERSION:
        raise SystemExit("parent full-run ledger protocol drifted")
    observed = {
        item["model_key"]: str(item["job_id"])
        for item in parent.get("evaluation_jobs", [])
    }
    if observed != PARENT_EVALUATIONS:
        raise SystemExit("parent evaluation job roster drifted")
    if str(parent.get("finalization_job", {}).get("job_id")) != PARENT_FINALIZER:
        raise SystemExit("parent finalizer identity drifted")
    if parent.get("evaluator_sha256") != sha256(FROZEN_EVALUATOR):
        raise SystemExit("frozen parent evaluator changed after submission")
    return manifest


def slurm_fields(job_id: str) -> dict[str, str]:
    try:
        output = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    except subprocess.CalledProcessError as error:
        accounting = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobIDRaw,State,Start",
            ],
            text=True,
        )
        for line in accounting.splitlines():
            if not line.strip():
                continue
            raw_id, state, start = line.split("|", 2)
            if raw_id == job_id:
                return {
                    "JobState": state.split()[0].split("+")[0],
                    "Reason": "AccountingOnly",
                    "Dependency": "(purged)",
                    "StartTime": start or "Unknown",
                }
        raise SystemExit(f"Slurm has no record of parent job {job_id}") from error
    fields: dict[str, str] = {}
    for token in output.replace("\n", " ").split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def validate_parent_jobs_held(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {job: {"state": "DRYRUN"} for job in HELD_JOBS}
    audit: dict[str, Any] = {}
    for job in HELD_JOBS:
        fields = slurm_fields(job)
        if fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser":
            raise SystemExit(
                f"sealed parent job {job} is not held: "
                f"state={fields.get('JobState')} reason={fields.get('Reason')}"
            )
        if fields.get("StartTime") not in {"Unknown", "N/A"}:
            raise SystemExit(f"sealed parent job {job} has a start time")
        audit[job] = {
            "state": fields.get("JobState"),
            "reason": fields.get("Reason"),
            "dependency": fields.get("Dependency"),
            "start_time": fields.get("StartTime"),
        }
    return audit


def validate_opening_amendment(path: Path) -> dict[str, Any]:
    amendment = json.loads(path.read_text(encoding="utf-8"))
    expected_jobs = list(PARENT_EVALUATIONS.values())
    if amendment.get("kind") != "holdout_opening_amendment":
        raise SystemExit("holdout opening amendment has the wrong kind")
    if amendment.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("holdout opening amendment protocol drifted")
    if amendment.get("parent_evaluation_jobs") != expected_jobs:
        raise SystemExit("holdout opening amendment parent job roster drifted")
    if str(amendment.get("parent_finalizer_job")) != PARENT_FINALIZER:
        raise SystemExit("holdout opening amendment parent finalizer drifted")
    if (
        amendment.get("holdout_will_open_before_qwen3_32b_full_training_completes")
        is not True
    ):
        raise SystemExit("holdout opening amendment does not disclose early opening")
    if amendment.get("qwen3_32b_will_run_regardless_of_parent_results") is not True:
        raise SystemExit("holdout opening amendment does not freeze the scale decision")
    if amendment.get("qwen3_32b_training_recipe_remains_frozen") is not True:
        raise SystemExit("holdout opening amendment does not freeze the recipe")
    return amendment


def validate_compatibility_amendment(path: Path) -> dict[str, Any]:
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("Qwen3-32B compatibility amendment protocol drifted")
    kind = amendment.get("kind")
    if kind == "zero3_compatibility_amendment":
        if str(amendment.get("failed_canary_job_id")) != "30885396":
            raise SystemExit("Qwen3-32B compatibility amendment failed-job drifted")
        if amendment.get("execution_only_changes") != [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
        ]:
            raise SystemExit("Qwen3-32B compatibility amendment flag set drifted")
    elif kind == "vllm_allreduce_compatibility_amendment":
        if str(amendment.get("failed_canary_job_id")) != "30892281":
            raise SystemExit("Qwen3-32B all-reduce amendment failed-job drifted")
        if amendment.get("execution_only_changes_added") != [
            "--disable-custom-all-reduce"
        ]:
            raise SystemExit("Qwen3-32B all-reduce amendment flag set drifted")
        if amendment.get("execution_only_changes_retained") != [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
        ]:
            raise SystemExit("Qwen3-32B retained compatibility flags drifted")
        superseded = Path(amendment.get("supersedes_compatibility_amendment", ""))
        if not superseded.is_file():
            raise SystemExit("Qwen3-32B superseded compatibility amendment is missing")
        if amendment.get("supersedes_compatibility_amendment_sha256") != sha256(
            superseded
        ):
            raise SystemExit("Qwen3-32B superseded amendment hash drifted")
        if amendment.get("corrected_shared_trainer_sha256") != sha256(SHARED_TRAINER):
            raise SystemExit("Qwen3-32B corrected shared trainer hash drifted")
    elif kind == "zero3_reentrant_checkpoint_compatibility_amendment":
        if str(amendment.get("failed_canary_job_id")) != "30908504":
            raise SystemExit("Qwen3-32B checkpoint amendment failed-job drifted")
        if amendment.get("execution_only_changes_added") != [
            "--gradient-checkpointing-use-reentrant"
        ]:
            raise SystemExit("Qwen3-32B checkpoint amendment flag set drifted")
        if amendment.get("execution_only_changes_retained") != [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
            "--disable-custom-all-reduce",
        ]:
            raise SystemExit("Qwen3-32B retained compatibility flags drifted")
        superseded = Path(amendment.get("supersedes_compatibility_amendment", ""))
        if not superseded.is_file():
            raise SystemExit("Qwen3-32B superseded compatibility amendment is missing")
        if amendment.get("supersedes_compatibility_amendment_sha256") != sha256(
            superseded
        ):
            raise SystemExit("Qwen3-32B superseded amendment hash drifted")
        if amendment.get("corrected_shared_trainer_sha256") != sha256(SHARED_TRAINER):
            raise SystemExit("Qwen3-32B corrected shared trainer hash drifted")
    elif kind == "zero3_adapter_artifact_contract_compatibility_amendment":
        if str(amendment.get("completed_canary_job_id")) != "30910721":
            raise SystemExit("Qwen3-32B completed-canary identity drifted")
        if str(amendment.get("failed_advance_job_id")) != "30910722":
            raise SystemExit("Qwen3-32B failed-advance identity drifted")
        if amendment.get("execution_only_changes") != [
            "write consolidated ZeRO-3 PEFT weights as adapter_model.safetensors while retaining adapter_model.bin",
            "strip one Markdown JSON fence before strict canary JSON parsing",
        ]:
            raise SystemExit("Qwen3-32B artifact-contract change set drifted")
        if amendment.get("reuses_completed_canary_without_retraining") is not True:
            raise SystemExit("Qwen3-32B artifact amendment must reuse the completed canary")
        if amendment.get("corrected_oat_deepspeed_sha256") != sha256(OAT_DEEPSPEED):
            raise SystemExit("Qwen3-32B corrected OAT serializer hash drifted")
        source = Path(amendment.get("source_adapter_bin", ""))
        converted = Path(amendment.get("converted_adapter_safetensors", ""))
        if not source.is_file() or amendment.get("source_adapter_bin_sha256") != sha256(source):
            raise SystemExit("Qwen3-32B source adapter artifact drifted")
        if (
            not converted.is_file()
            or amendment.get("converted_adapter_safetensors_sha256")
            != sha256(converted)
        ):
            raise SystemExit("Qwen3-32B converted adapter artifact drifted")
        coverage = amendment.get("canary_parse_audit", {})
        if coverage != {
            "draws": 2560,
            "strict_json_before_fence_normalization": 0,
            "single_fenced_json": 2560,
            "invalid_after_fence_normalization": 0,
        }:
            raise SystemExit("Qwen3-32B canary parse audit drifted")
        superseded = Path(amendment.get("supersedes_compatibility_amendment", ""))
        if not superseded.is_file():
            raise SystemExit("Qwen3-32B superseded compatibility amendment is missing")
        if amendment.get("supersedes_compatibility_amendment_sha256") != sha256(
            superseded
        ):
            raise SystemExit("Qwen3-32B superseded amendment hash drifted")
    else:
        raise SystemExit("Qwen3-32B compatibility amendment has the wrong kind")
    if amendment.get("scientific_hyperparameters_changed") is not False:
        raise SystemExit("Qwen3-32B compatibility amendment changes the estimand")
    if kind != "zero3_adapter_artifact_contract_compatibility_amendment":
        replacement = amendment.get("replacement_canary", {})
        if replacement != {
            "seed": 42,
            "max_train": 160,
            "eval_steps": 10,
            "performance_threshold": False,
        }:
            raise SystemExit("Qwen3-32B replacement-canary specification drifted")
    if amendment.get("corrected_trainer_sha256") != sha256(TRAIN):
        raise SystemExit("Qwen3-32B corrected trainer hash drifted")
    if amendment.get("corrected_launcher_sha256") != sha256(Path(__file__)):
        raise SystemExit("Qwen3-32B corrected launcher hash drifted")
    return amendment


def audit_parent_jobs(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {job: {"state": "DRYRUN"} for job in HELD_JOBS}
    audit: dict[str, Any] = {}
    for job in HELD_JOBS:
        fields = slurm_fields(job)
        audit[job] = {
            "state": fields.get("JobState"),
            "reason": fields.get("Reason"),
            "dependency": fields.get("Dependency"),
            "start_time": fields.get("StartTime"),
        }
    return audit


def validate_static_inputs(
    *, dry_run: bool, opening_amendment: Path | None = None
) -> dict[str, Any]:
    if opening_amendment is None:
        parent_jobs = validate_parent_jobs_held(dry_run=dry_run)
        opening = None
    else:
        opening = validate_opening_amendment(opening_amendment)
        parent_jobs = audit_parent_jobs(dry_run=dry_run)
    return {
        "holdout_manifest": validate_parent_holdout(),
        "model_cache": validate_model_cache(),
        "parent_jobs": parent_jobs,
        "opening_amendment": opening,
        "code_sha256": code_hashes(),
    }


def submit(command: list[str], dry_run: bool) -> str:
    print(shlex.join(command), flush=True)
    if dry_run:
        return "DRYRUN"
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def route_to_registered_account(
    job_id: str, args: argparse.Namespace, *, dry_run: bool
) -> dict[str, str]:
    if dry_run:
        return {"job_id": job_id, "account": args.account, "partition": args.partition}
    subprocess.check_call(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"Account={args.account}",
            f"Partition={args.partition}",
        ]
    )
    fields = slurm_fields(job_id)
    if (
        fields.get("Account") != args.account
        or fields.get("Partition") != args.partition
    ):
        raise SystemExit(f"job {job_id} was not routed to mltheory/all")
    return {
        "job_id": job_id,
        "account": fields["Account"],
        "partition": fields["Partition"],
    }


def write_ledger(kind: str, payload: dict[str, Any]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"exp4_qwen32_{kind}_{timestamp}.json"
    document = {
        "kind": kind,
        "protocol_version": PROTOCOL_VERSION,
        "registered_at" if kind == "registration" else "recorded_at": timestamp,
        **payload,
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ledger -> {path}", flush=True)
    return path


def validate_registration(path: Path, *, dry_run: bool) -> dict[str, Any]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration.get("kind") != "registration":
        raise SystemExit("registration ledger has the wrong kind")
    if registration.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("registration protocol version drifted")
    opening_path_text = registration.get("holdout_opening_amendment")
    opening_path = Path(opening_path_text) if opening_path_text else None
    static = validate_static_inputs(dry_run=dry_run, opening_amendment=opening_path)
    if registration.get("protocol_sha256") != sha256(PROTOCOL):
        raise SystemExit("Qwen3-32B protocol changed after registration")
    if registration.get("code_sha256") != static["code_sha256"]:
        raise SystemExit("Qwen3-32B implementation changed after registration")
    if registration.get("holdout_manifest_sha256") != sha256(DATA / "manifest.json"):
        raise SystemExit("holdout manifest changed after Qwen3-32B registration")
    if registration.get("holdout_tasks_sha256") != sha256(DATA / "test.tasks.jsonl"):
        raise SystemExit("holdout tasks changed after Qwen3-32B registration")
    if opening_path is not None:
        if registration.get("holdout_opening_amendment_sha256") != sha256(opening_path):
            raise SystemExit("holdout opening amendment changed after registration")
        canary_registration_text = registration.get("canary_registration_ledger")
        if not canary_registration_text:
            raise SystemExit(
                "amended registration does not bind the canary registration"
            )
        canary_registration = Path(canary_registration_text)
        if registration.get("canary_registration_ledger_sha256") != sha256(
            canary_registration
        ):
            raise SystemExit("canary registration changed after amended registration")
    compatibility_text = registration.get("compatibility_amendment")
    if compatibility_text:
        compatibility_path = Path(compatibility_text)
        validate_compatibility_amendment(compatibility_path)
        if registration.get("compatibility_amendment_sha256") != sha256(
            compatibility_path
        ):
            raise SystemExit("compatibility amendment changed after registration")
    return registration


def register(args: argparse.Namespace) -> Path:
    opening_path = (
        args.holdout_opening_amendment.resolve()
        if args.holdout_opening_amendment
        else None
    )
    canary_registration_path = (
        args.canary_registration_ledger.resolve()
        if args.canary_registration_ledger
        else None
    )
    compatibility_path = (
        args.compatibility_amendment.resolve() if args.compatibility_amendment else None
    )
    if (opening_path is None) != (canary_registration_path is None):
        raise SystemExit(
            "opening amendment and canary registration ledger must be supplied together"
        )
    static = validate_static_inputs(
        dry_run=args.dry_run, opening_amendment=opening_path
    )
    if compatibility_path is not None:
        validate_compatibility_amendment(compatibility_path)
    manifest = static["holdout_manifest"]
    return write_ledger(
        "registration",
        {
            "dry_run": args.dry_run,
            "supersedes_failed_registration": (
                str(args.supersedes_registration.resolve())
                if args.supersedes_registration
                else None
            ),
            "decision": "run three dense Qwen3-32B seeds; retain Qwen3-14B; exclude Qwen3-30B-A3B",
            "protocol_sha256": sha256(PROTOCOL),
            "parent_protocol_sha256": sha256(PARENT_PROTOCOL),
            "parent_ledger": str(PARENT_LEDGER),
            "parent_ledger_sha256": sha256(PARENT_LEDGER),
            "holdout_manifest_sha256": sha256(DATA / "manifest.json"),
            "holdout_tasks_sha256": sha256(DATA / "test.tasks.jsonl"),
            "holdout_public_description": manifest["holdout"],
            "model_cache": static["model_cache"],
            "parent_jobs_at_registration": static["parent_jobs"],
            "holdout_opening_amendment": (str(opening_path) if opening_path else None),
            "holdout_opening_amendment_sha256": (
                sha256(opening_path) if opening_path else None
            ),
            "canary_registration_ledger": (
                str(canary_registration_path) if canary_registration_path else None
            ),
            "canary_registration_ledger_sha256": (
                sha256(canary_registration_path) if canary_registration_path else None
            ),
            "compatibility_amendment": (
                str(compatibility_path) if compatibility_path else None
            ),
            "compatibility_amendment_sha256": (
                sha256(compatibility_path) if compatibility_path else None
            ),
            "code_sha256": static["code_sha256"],
            "seeds": list(SEEDS),
            "canary_gate": "feasibility_only_no_performance_threshold",
            "scientific_hyperparameters_changed_from_qwen3_14b": False,
            "execution_changes": {
                "a6000_gpus": 4,
                "rollout_tensor_parallel": 2,
                "scheduler_route": {
                    "submission_account": args.submit_account,
                    "submission_partition": args.submit_partition,
                    "submission_qos": args.submit_qos,
                    "registered_account": args.account,
                    "registered_partition": args.partition,
                },
                "learner_world_size": 2,
                "zero_stage": 3,
                "rollout_batch_size_per_learner": 8,
                "policy_buffer_per_learner": 64,
            },
        },
    )


def canary_command(args: argparse.Namespace) -> list[str]:
    env = {
        "MODEL_KEY": MODEL_KEY,
        "MODEL": MODEL,
        "PROMPT_TEMPLATE": "auto_no_think",
        "SEED": "42",
        "RUN_KIND": "canary",
        "MAX_TRAIN": "160",
        "EVAL_STEPS": "10",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        f"--account={args.submit_account}",
        f"--partition={args.submit_partition}",
        f"--qos={args.submit_qos}",
        "--gres=gpu:a6000:4",
        "--mem=220G",
        "--cpus-per-task=12",
        f"--time={args.canary_walltime}",
        f"--exclude={args.exclude}",
        f"--export={export}",
        str(TRAIN),
    ]


def launch_canary(args: argparse.Namespace) -> None:
    registration_path = args.registration_ledger.resolve()
    validate_registration(registration_path, dry_run=args.dry_run)
    command = canary_command(args)
    canary_job = submit(command, args.dry_run)
    canary_route = route_to_registered_account(canary_job, args, dry_run=args.dry_run)
    canary_ledger = write_ledger(
        "canary",
        {
            "dry_run": args.dry_run,
            "registration_ledger": str(registration_path),
            "registration_ledger_sha256": sha256(registration_path),
            "job_id": canary_job,
            "scheduler_route": canary_route,
            "command": command,
        },
    )

    env = {
        "REGISTRATION_LEDGER": str(registration_path),
        "CANARY_LEDGER": str(canary_ledger),
        "CANARY_JOB_ID": canary_job,
        "QWEN32_ACCOUNT": args.account,
        "QWEN32_PARTITION": args.partition,
        "QWEN32_SUBMIT_ACCOUNT": args.submit_account,
        "QWEN32_SUBMIT_PARTITION": args.submit_partition,
        "QWEN32_SUBMIT_QOS": args.submit_qos,
        "QWEN32_EXCLUDE": args.exclude,
        "QWEN32_TRAIN_WALLTIME": args.walltime,
        "QWEN32_EVAL_WALLTIME": args.eval_walltime,
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    advance_command = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        "--time=00:30:00",
        "--mem=4G",
        "--cpus-per-task=1",
        f"--exclude={args.exclude}",
        f"--dependency=afterok:{canary_job}",
        f"--export={export}",
        str(ADVANCE),
    ]
    advance_job = submit(advance_command, args.dry_run)
    write_ledger(
        "advance",
        {
            "dry_run": args.dry_run,
            "registration_ledger": str(registration_path),
            "canary_ledger": str(canary_ledger),
            "canary_job_id": canary_job,
            "job_id": advance_job,
            "environment": env,
            "command": advance_command,
        },
    )
    print(f"submitted Qwen3-32B canary {canary_job}; automatic advance {advance_job}")


def job_state(job_id: str) -> str:
    output = subprocess.check_output(
        ["sacct", "-j", job_id, "--noheader", "-X", "--format=State"], text=True
    )
    states = [
        line.strip().split()[0].split("+")[0]
        for line in output.splitlines()
        if line.strip()
    ]
    if not states:
        raise SystemExit(f"Slurm has no record of canary job {job_id}")
    return states[0]


def parse_probability(text: str) -> float | None:
    candidate = text.strip()
    lines = candidate.splitlines()
    fence = chr(96) * 3
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {fence, fence + "json"}
        and lines[-1].strip() == fence
    ):
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)["yes_prob"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def validate_canary(job_id: str) -> dict[str, Any]:
    state = job_state(job_id)
    if state != "COMPLETED":
        raise SystemExit(f"Qwen3-32B canary {job_id} is {state}, not COMPLETED")
    roots = sorted(REPORTS.glob(f"qwen32_canary_{MODEL_KEY}_market_*_j{job_id}"))
    if len(roots) != 1:
        raise SystemExit(f"expected one Qwen3-32B canary root, got {roots}")
    root = roots[0]
    adapters = sorted(root.glob("debug_*/saved_models/step_*"))
    if not adapters or not (adapters[-1] / "adapter_model.safetensors").is_file():
        raise SystemExit("Qwen3-32B canary did not save a valid adapter")
    eval_files = sorted(
        root.glob("debug_*/eval_results/*.json"), key=lambda path: int(path.stem)
    )
    if not eval_files:
        raise SystemExit("Qwen3-32B canary did not save development evaluation")
    rows = json.loads(eval_files[-1].read_text(encoding="utf-8"))
    if len(rows) != 512:
        raise SystemExit("Qwen3-32B canary development evaluation is incomplete")
    parsed = [
        parse_probability(output) for row in rows for output in row.get("output", [])
    ]
    if len(parsed) != 512 * 5 or any(value is None for value in parsed):
        raise SystemExit("Qwen3-32B canary failed full development parse coverage")
    scores = [float(score) for row in rows for score in row.get("scores", [])]
    if len(scores) != 512 * 5 or any(not math.isfinite(score) for score in scores):
        raise SystemExit("Qwen3-32B canary scores are incomplete or nonfinite")
    return {
        "job_id": job_id,
        "state": state,
        "report": str(root),
        "final_eval": str(eval_files[-1]),
        "adapter_sha256": sha256(adapters[-1] / "adapter_model.safetensors"),
        "development_parse_coverage": 1.0,
    }


def validate_canary_ledger(
    path: Path,
    registration_path: Path,
    job_id: str,
    registration: dict[str, Any],
) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("kind") != "canary" or str(ledger.get("job_id")) != job_id:
        raise SystemExit("Qwen3-32B canary ledger identity drifted")
    permitted_registrations = {registration_path.resolve()}
    legacy_registration = registration.get("canary_registration_ledger")
    if legacy_registration:
        permitted_registrations.add(Path(legacy_registration).resolve())
    ledger_registration_text = ledger.get("registration_ledger")
    if not ledger_registration_text:
        raise SystemExit("Qwen3-32B canary ledger registration is missing")
    canary_registration = Path(ledger_registration_text).resolve()
    if canary_registration not in permitted_registrations:
        raise SystemExit("Qwen3-32B canary ledger registration drifted")
    if ledger.get("registration_ledger_sha256") != sha256(canary_registration):
        raise SystemExit("Qwen3-32B registration changed after canary submission")
    if ledger.get("dry_run") is not False:
        raise SystemExit("Qwen3-32B full launch requires a real canary")
    return ledger


def training_command(
    args: argparse.Namespace, seed: int
) -> tuple[list[str], dict[str, str]]:
    env = {
        "MODEL_KEY": MODEL_KEY,
        "MODEL": MODEL,
        "PROMPT_TEMPLATE": "auto_no_think",
        "SEED": str(seed),
        "RUN_KIND": "train",
        "MAX_TRAIN": "4800",
        "EVAL_STEPS": "25",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch",
        "--parsable",
        f"--account={args.submit_account}",
        f"--partition={args.submit_partition}",
        f"--qos={args.submit_qos}",
        "--gres=gpu:a6000:4",
        "--mem=220G",
        "--cpus-per-task=12",
        f"--time={args.walltime}",
        f"--exclude={args.exclude}",
        f"--export={export}",
        str(TRAIN),
    ]
    return command, env


def launch_full(args: argparse.Namespace) -> None:
    registration_path = args.registration_ledger.resolve()
    canary_ledger_path = args.canary_ledger.resolve()
    registration = validate_registration(registration_path, dry_run=args.dry_run)
    if not args.dry_run:
        validate_canary_ledger(
            canary_ledger_path,
            registration_path,
            args.canary_job_id,
            registration,
        )
        canary = validate_canary(args.canary_job_id)
    else:
        canary = {"job_id": args.canary_job_id, "state": "DRYRUN"}

    training_jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        command, env = training_command(args, seed)
        job_id = submit(command, args.dry_run)
        scheduler_route = route_to_registered_account(
            job_id, args, dry_run=args.dry_run
        )
        training_jobs.append(
            {
                "seed": seed,
                "job_id": job_id,
                "scheduler_route": scheduler_route,
                "environment": env,
                "command": command,
            }
        )
    dependencies = ":".join(item["job_id"] for item in training_jobs)
    adapter_specs = ";".join(
        f"{item['seed']}:job={item['job_id']}" for item in training_jobs
    )

    eval_env = {
        "MODEL_KEY": MODEL_KEY,
        "MODEL": MODEL,
        "PROMPT_TEMPLATE": "auto_no_think",
        "ADAPTER_SPECS": adapter_specs,
    }
    eval_export = "ALL," + ",".join(f"{key}={value}" for key, value in eval_env.items())
    eval_command = [
        "sbatch",
        "--parsable",
        f"--account={args.submit_account}",
        f"--partition={args.submit_partition}",
        f"--qos={args.submit_qos}",
        "--gres=gpu:a6000:2",
        "--mem=140G",
        "--cpus-per-task=10",
        f"--time={args.eval_walltime}",
        f"--exclude={args.exclude}",
        f"--dependency=afterok:{dependencies}",
        f"--export={eval_export}",
        str(LOCKED),
    ]
    evaluation_job = submit(eval_command, args.dry_run)
    evaluation_route = route_to_registered_account(
        evaluation_job, args, dry_run=args.dry_run
    )

    if registration.get("holdout_opening_amendment"):
        release_record: dict[str, Any] = {
            "status": "not_submitted_parent_holdout_already_opened",
            "holdout_opening_amendment": registration["holdout_opening_amendment"],
        }
    else:
        held_ids = ":".join(HELD_JOBS)
        release_env = {"HELD_JOB_IDS": held_ids}
        release_command = [
            "sbatch",
            "--parsable",
            f"--account={args.account}",
            f"--partition={args.partition}",
            "--time=00:10:00",
            "--mem=2G",
            "--cpus-per-task=1",
            f"--exclude={args.exclude}",
            f"--dependency=afterok:{dependencies}",
            f"--export=ALL,HELD_JOB_IDS={held_ids}",
            str(RELEASE),
        ]
        release_job = submit(release_command, args.dry_run)
        release_record = {
            "status": "submitted",
            "job_id": release_job,
            "environment": release_env,
            "command": release_command,
        }

    evaluation_specs = ";".join(
        [
            *(f"{key}:{job}" for key, job in PARENT_EVALUATIONS.items()),
            f"{MODEL_KEY}:{evaluation_job}",
        ]
    )
    evaluation_dependencies = ":".join([*PARENT_EVALUATIONS.values(), evaluation_job])
    final_env = {
        "EVALUATION_SPECS": evaluation_specs,
        "REGISTRATION_LEDGER": str(registration_path),
    }
    final_export = "ALL," + ",".join(
        f"{key}={value}" for key, value in final_env.items()
    )
    final_command = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        "--time=01:00:00",
        "--mem=16G",
        "--cpus-per-task=2",
        f"--exclude={args.exclude}",
        f"--dependency=afterok:{evaluation_dependencies}",
        f"--export={final_export}",
        str(FINALIZER),
    ]
    finalizer_job = submit(final_command, args.dry_run)

    write_ledger(
        "full",
        {
            "dry_run": args.dry_run,
            "registration_ledger": str(registration_path),
            "registration_ledger_sha256": sha256(registration_path),
            "canary": canary,
            "canary_ledger": str(canary_ledger_path),
            "canary_ledger_sha256": sha256(canary_ledger_path),
            "training_jobs": training_jobs,
            "qwen3_32b_evaluation": {
                "job_id": evaluation_job,
                "scheduler_route": evaluation_route,
                "environment": eval_env,
                "command": eval_command,
            },
            "parent_evaluations": PARENT_EVALUATIONS,
            "parent_finalizer": PARENT_FINALIZER,
            "holdout_opening_amendment": registration.get("holdout_opening_amendment"),
            "release_job": release_record,
            "combined_finalizer": {
                "job_id": finalizer_job,
                "environment": final_env,
                "command": final_command,
            },
        },
    )
    print(
        f"submitted three Qwen3-32B seeds, registered evaluation {evaluation_job}, "
        f"and combined finalizer {finalizer_job}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("register", "canary", "full"), required=True
    )
    parser.add_argument("--registration-ledger", type=Path)
    parser.add_argument("--canary-ledger", type=Path)
    parser.add_argument("--canary-job-id", default="")
    parser.add_argument("--supersedes-registration", type=Path)
    parser.add_argument("--holdout-opening-amendment", type=Path)
    parser.add_argument("--canary-registration-ledger", type=Path)
    parser.add_argument("--compatibility-amendment", type=Path)
    parser.add_argument("--account", default="mltheory")
    parser.add_argument("--partition", default="all")
    parser.add_argument("--submit-account", default="allcs")
    parser.add_argument("--submit-partition", default="cs")
    parser.add_argument("--submit-qos", default="medium")
    parser.add_argument("--exclude", default="node206")
    parser.add_argument("--canary-walltime", default="04:00:00")
    parser.add_argument("--walltime", default="30:00:00")
    parser.add_argument("--eval-walltime", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.account != "mltheory" or args.partition != "all":
        parser.error(
            "Qwen3-32B extension is registered to account mltheory on partition all"
        )
    if (
        args.submit_account != "allcs"
        or args.submit_partition != "cs"
        or args.submit_qos != "medium"
    ):
        parser.error(
            "Qwen3-32B scheduler bootstrap is locked to allcs/cs with medium QOS"
        )
    if args.stage in {"canary", "full"} and args.registration_ledger is None:
        parser.error(f"--stage {args.stage} requires --registration-ledger")
    if args.stage == "full" and (args.canary_ledger is None or not args.canary_job_id):
        parser.error("--stage full requires --canary-ledger and --canary-job-id")

    if args.stage == "register":
        print(f"registered -> {register(args)}")
    elif args.stage == "canary":
        launch_canary(args)
    else:
        launch_full(args)


if __name__ == "__main__":
    main()
