#!/usr/bin/env python3
"""Canary-gated launcher for the registered Qwen3-14B Coin City extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "QWEN3_14B_SCALE_PROTOCOL.md"
PARENT_MANIFEST = ROOT / "protocol" / "coin_city_structural_manifest.json"
PARENT_RESULTS = REPORTS / "registered_results.json"
TRAIN = ROOT / "train.sh"
BASE_EVAL = ROOT / "base_eval.sh"
REPORT = ROOT / "report.py"
CORE_ANALYSIS = ROOT / "make_paper_outputs.py"
RENDERER = ROOT / "render_qwen3_14b_extension.py"
ADVANCE = ROOT / "advance_qwen3_14b_extension.sbatch"
FINALIZER = ROOT / "finalize_qwen3_14b_extension.sbatch"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"
PROTOCOL_VERSION = "coin_city_qwen3_14b_scale_v1"
MODEL_KEY = "qwen3_14b"
MODEL = "Qwen/Qwen3-14B"
TEMPLATE = "auto_no_think"
ARMS = ("causal", "population_prior")
SEEDS = (42, 43, 44)
REWARD_RE = re.compile(r"['\"]actor/rewards['\"]\s*:\s*([^,\s}\]]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def validate_model_cache() -> str:
    cache = HF_HUB / ("models--" + MODEL.replace("/", "--"))
    snapshots = sorted((cache / "snapshots").glob("*")) if cache.is_dir() else []
    complete = [snapshot for snapshot in snapshots if complete_model_snapshot(snapshot)]
    if len(complete) != 1:
        raise SystemExit(
            f"expected exactly one complete offline {MODEL} snapshot, found {complete}"
        )
    return complete[0].name


def validate_inputs() -> dict[str, Any]:
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "coin_city_structural_transfer_v1":
        raise SystemExit("unexpected original Coin City protocol")
    if manifest.get("status") != "passed" or not all(manifest.get("checks", {}).values()):
        raise SystemExit("original Coin City preflight is not fully passed")
    design = manifest.get("registered_design", {})
    if design.get("heldout_rows_per_checkpoint") != 1440:
        raise SystemExit("original held-out task count drifted")
    if design.get("train_rows_per_run") != 4800:
        raise SystemExit("original training count drifted")
    if design.get("training_seeds") != list(SEEDS):
        raise SystemExit("original seed roster drifted")
    if design.get("confirmatory_arms") != list(ARMS):
        raise SystemExit("original reward-arm roster drifted")

    frozen_dataset_sha256 = {}
    for relative, expected in manifest.get("dataset_sha256", {}).items():
        if Path(relative).name.startswith("cache-"):
            continue
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen Coin City dataset drifted: {relative}")
        frozen_dataset_sha256[relative] = expected
    for name, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen Coin City source drifted: {name}")

    parent = json.loads(PARENT_RESULTS.read_text(encoding="utf-8"))
    if parent.get("protocol") != "coin_city_structural_transfer_v1":
        raise SystemExit("canonical original result has the wrong protocol")
    if parent.get("validated_jobs") != 24 or parent.get("row_draws") != 207_360:
        raise SystemExit("canonical original result has an incomplete roster")
    parent_ledger = REPO / parent["ledger"]
    if sha256(parent_ledger) != parent.get("ledger_sha256"):
        raise SystemExit("canonical original ledger drifted")
    qwen8_files = {
        relative: parent["score_file_sha256"][relative]
        for relative in parent.get("score_files", [])
        if "qwen3_8b" in relative
    }
    if len(qwen8_files) != 14:
        raise SystemExit("canonical original result lacks 14 Qwen3-8B endpoint files")
    for relative, expected in qwen8_files.items():
        path = REPO / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"frozen Qwen3-8B endpoint drifted: {relative}")

    snapshot_id = validate_model_cache()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256(PROTOCOL),
        "parent_manifest_sha256": sha256(PARENT_MANIFEST),
        "parent_results_sha256": sha256(PARENT_RESULTS),
        "parent_ledger": str(parent_ledger),
        "parent_ledger_sha256": sha256(parent_ledger),
        "parent_qwen8_score_sha256": qwen8_files,
        "frozen_dataset_sha256": frozen_dataset_sha256,
        "model_snapshot": snapshot_id,
        "train_script_sha256": sha256(TRAIN),
        "base_eval_script_sha256": sha256(BASE_EVAL),
        "report_script_sha256": sha256(REPORT),
        "core_analysis_sha256": sha256(CORE_ANALYSIS),
        "renderer_sha256": sha256(RENDERER),
        "advance_script_sha256": sha256(ADVANCE),
        "finalizer_script_sha256": sha256(FINALIZER),
        "launcher_sha256": sha256(Path(__file__).resolve()),
    }


def ensure_new_extension() -> None:
    full_ledgers = sorted(RUNS.glob("coin_city_qwen3_14b_scale_full_*.json"))
    scientific_reports = sorted(REPORTS.glob("scale_causal_qwen3_14b_s*_j*"))
    scientific_reports += sorted(
        REPORTS.glob("scale_population_prior_qwen3_14b_s*_j*")
    )
    base_reports = sorted(REPORTS.glob("base_qwen3_14b_j*"))
    if full_ledgers or scientific_reports or base_reports:
        raise SystemExit(
            "Qwen3-14B extension already has scientific state; refusing duplicate launch"
        )


def submit(command: list[str], dry_run: bool) -> str:
    print(shlex.join(command), flush=True)
    if dry_run:
        return "DRYRUN"
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_ledger(kind: str, payload: dict[str, Any]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"coin_city_qwen3_14b_scale_{kind}_{timestamp()}.json"
    payload = {**payload, "ledger_name": path.name}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ledger -> {path}")
    return path


def rewrite_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_env(arm: str, seed: int, *, canary: bool, vllm_ratio: float) -> dict[str, str]:
    tag_prefix = "scale_canary_" if canary else "scale_"
    return {
        "ARM": arm,
        "BATCH": "16",
        "EVAL_BATCH": "120",
        "EVAL_STEPS": "10" if canary else "50",
        "GEN_LEN": "192",
        "GPUS": "2",
        "LORA_ALPHA": "64",
        "LORA_RANK": "32",
        "LR": "0.000001",
        "MAX_MODEL_LEN": "3072",
        "MAX_TRAIN": "160" if canary else "4800",
        "MODEL": MODEL,
        "MODEL_KEY": MODEL_KEY,
        "PROMPT_MAX_LENGTH": "2304",
        "PROMPT_TEMPLATE": TEMPLATE,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RESPONSE_W": "0.60",
        "ROLLOUT_PER_PROMPT": "8",
        "SEED": str(seed),
        "STOCHASTIC_N": "5",
        "STOCHASTIC_TEMP": "0.7",
        "TAG": f"{tag_prefix}{arm}_{MODEL_KEY}_s{seed}",
        "TEMP": "1.3",
        "VLLM_RATIO": str(vllm_ratio),
        "ZERO_STAGE": "2",
    }


def sbatch_train(
    env: dict[str, str], *, account: str, partition: str, exclude: str, walltime: str
) -> list[str]:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--partition={partition}",
        "--gres=gpu:a6000:2",
        "--cpus-per-task=8",
        "--mem=100G",
        f"--time={walltime}",
        f"--exclude={exclude}",
        f"--export={export}",
        str(TRAIN),
    ]


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


def parse_rate(path: Path) -> tuple[float, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    draws = sum(int(row["n_draws"]) for row in rows)
    parsed = sum(float(row["parse_rate"]) * int(row["n_draws"]) for row in rows)
    return parsed / draws, draws


def endpoint_contract(path: Path, *, temperature: float, draws: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != 1440:
        raise SystemExit(f"{path}: expected 1,440 records, found {len(records)}")
    for index, record in enumerate(records):
        outputs = record.get("output")
        if not isinstance(outputs, list) or len(outputs) != draws:
            raise SystemExit(f"{path}: record {index} has the wrong draw count")
        if float(record.get("temperature", -1)) != temperature:
            raise SystemExit(f"{path}: record {index} has the wrong temperature")
        if record.get("structured_output") != "forecast_array":
            raise SystemExit(f"{path}: record {index} lacks the registered grammar")


def validate_canary(job_id: str) -> dict[str, Any]:
    state = job_state(job_id)
    if state != "COMPLETED":
        raise SystemExit(f"canary job {job_id} is {state}, not COMPLETED")
    roots = sorted(REPORTS.glob(f"scale_canary_causal_{MODEL_KEY}_s42_*_j{job_id}"))
    if len(roots) != 1:
        raise SystemExit(f"expected one canary report root for {job_id}, found {roots}")
    root = roots[0]
    artifacts = {
        "greedy_raw": root / "greedy.json",
        "greedy_summary": root / "greedy.scores.summary.json",
        "stochastic_raw": root / "stochastic_n5.json",
        "stochastic_summary": root / "stochastic_n5.scores.summary.json",
    }
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"canary is missing {path}")
    endpoint_contract(artifacts["greedy_raw"], temperature=0.0, draws=1)
    endpoint_contract(artifacts["stochastic_raw"], temperature=0.7, draws=5)
    greedy_rate, greedy_draws = parse_rate(artifacts["greedy_summary"])
    stochastic_rate, stochastic_draws = parse_rate(artifacts["stochastic_summary"])
    if greedy_draws != 1440 or stochastic_draws != 7200:
        raise SystemExit("canary endpoint draw counts are incomplete")
    if min(greedy_rate, stochastic_rate) < 0.95:
        raise SystemExit("canary parse coverage is below 95 percent")
    adapters = list(root.glob("debug_*/saved_models/step_*/adapter_model.safetensors"))
    if len(adapters) != 1 or adapters[0].stat().st_size == 0:
        raise SystemExit(f"canary expected one saved adapter, found {adapters}")
    rewards = [float(value) for value in REWARD_RE.findall((root / "train.log").read_text(errors="replace"))]
    if len(rewards) < 2 or not np.isfinite(rewards).all() or np.ptp(rewards) < 1e-4:
        raise SystemExit("canary training rewards are missing, nonfinite, or constant")
    return {
        "job_id": job_id,
        "state": state,
        "report": str(root),
        "adapter_sha256": sha256(adapters[0]),
        "greedy_parse_rate": greedy_rate,
        "stochastic_parse_rate": stochastic_rate,
        "reward_range": float(np.ptp(rewards)),
    }


def validate_canary_ledger(path: Path, job_id: str, current: dict[str, Any]) -> dict:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if str(ledger.get("canary_job_id")) != job_id:
        raise SystemExit("canary ledger job ID does not match")
    if ledger.get("dry_run") is not False:
        raise SystemExit("canary ledger is not a real submission")
    for key, value in current.items():
        if ledger.get(key) != value:
            raise SystemExit(f"registered canary input drifted: {key}")
    return ledger


def launch_canary(args: argparse.Namespace) -> None:
    registration = validate_inputs()
    ensure_new_extension()
    prior_canaries = sorted(RUNS.glob("coin_city_qwen3_14b_scale_canary_*.json"))
    if prior_canaries:
        raise SystemExit(f"registered 14B canary already exists: {prior_canaries}")
    env = train_env("causal", 42, canary=True, vllm_ratio=args.vllm_ratio)
    command = sbatch_train(
        env,
        account=args.account,
        partition=args.partition,
        exclude=args.exclude,
        walltime=args.canary_walltime,
    )
    job_id = submit(command, args.dry_run)
    ledger_payload = {
        **registration,
        "submitted_at": timestamp(),
        "dry_run": args.dry_run,
        "canary_job_id": job_id,
        "canary_environment": env,
        "canary_command": command,
        "canary_scientific_status": "excluded",
    }
    ledger_path = write_ledger("canary", ledger_payload)
    ledger_payload["ledger_name"] = ledger_path.name

    advance_env = {
        "CANARY_JOB_ID": job_id,
        "SCALE_ACCOUNT": args.account,
        "CANARY_LEDGER": str(ledger_path),
        "SCALE_PARTITION": args.partition,
        "SCALE_EXCLUDE": args.exclude,
        "SCALE_VLLM_RATIO": str(args.vllm_ratio),
        "SCALE_TRAIN_WALLTIME": args.walltime,
        "SCALE_BASE_WALLTIME": args.base_walltime,
        "SCALE_FINALIZE_WALLTIME": args.finalize_walltime,
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in advance_env.items())
    advance_command = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        "--cpus-per-task=1",
        "--mem=4G",
        f"--time={args.advance_walltime}",
        f"--exclude={args.exclude}",
        f"--dependency=afterok:{job_id}",
        f"--export={export}",
        str(ADVANCE),
    ]
    advance_job_id = submit(advance_command, args.dry_run)
    ledger_payload["advance_job"] = {
        "job_id": advance_job_id,
        "environment": advance_env,
        "command": advance_command,
        "dependency": f"afterok:{job_id}",
    }
    rewrite_ledger(ledger_path, ledger_payload)
    print(f"submitted Qwen3-14B canary {job_id} and gated advance {advance_job_id}")


def launch_full(args: argparse.Namespace) -> None:
    registration = validate_inputs()
    ensure_new_extension()
    canary_ledger = validate_canary_ledger(args.canary_ledger, args.canary_job_id, registration)
    canary_audit = (
        {"job_id": args.canary_job_id, "state": "DRYRUN"}
        if args.dry_run
        else validate_canary(args.canary_job_id)
    )

    training_jobs: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            env = train_env(arm, seed, canary=False, vllm_ratio=args.vllm_ratio)
            command = sbatch_train(
                env,
                account=args.account,
                partition=args.partition,
                exclude=args.exclude,
                walltime=args.walltime,
            )
            job_id = submit(command, args.dry_run)
            training_jobs.append(
                {"model": MODEL_KEY, "arm": arm, "seed": seed, "job_id": job_id,
                 "environment": env, "command": command}
            )

    base_env = {"MODEL_KEY": MODEL_KEY, "MODEL": MODEL, "PROMPT_TEMPLATE": TEMPLATE}
    base_export = "ALL," + ",".join(f"{key}={value}" for key, value in base_env.items())
    base_command = [
        "sbatch", "--parsable", f"--account={args.account}", f"--partition={args.partition}",
        "--gres=gpu:a6000:1", "--cpus-per-task=8", "--mem=48G",
        f"--time={args.base_walltime}", f"--exclude={args.exclude}",
        f"--export={base_export}", str(BASE_EVAL),
    ]
    base_job_id = submit(base_command, args.dry_run)
    base_evaluation = {
        "model": MODEL_KEY,
        "job_id": base_job_id,
        "environment": base_env,
        "command": base_command,
    }
    payload = {
        **registration,
        "submitted_at": timestamp(),
        "dry_run": args.dry_run,
        "canary_ledger": str(args.canary_ledger),
        "canary_ledger_sha256": sha256(args.canary_ledger),
        "canary_audit": canary_audit,
        "training_jobs": training_jobs,
        "base_evaluation": base_evaluation,
        "counts": {"confirmatory_training": 6, "base_evaluations": 1,
                   "scientific_jobs": 7, "endpoint_files": 14,
                   "scored_row_draws": 60_480},
    }
    ledger_path = write_ledger("full", payload)
    payload["ledger_name"] = ledger_path.name

    scientific_ids = [row["job_id"] for row in training_jobs] + [base_job_id]
    dependencies = ":".join(job_id for job_id in scientific_ids if job_id != "DRYRUN")
    finalizer_env = {
        "EXTENSION_LEDGER": str(ledger_path),
        "EXPECTED_RENDERER_SHA256": registration["renderer_sha256"],
        "EXPECTED_PROTOCOL_SHA256": registration["protocol_sha256"],
    }
    finalizer_export = "ALL," + ",".join(
        f"{key}={value}" for key, value in finalizer_env.items()
    )
    finalizer_command = [
        "sbatch", "--parsable", f"--account={args.account}",
        f"--partition={args.partition}",
        "--cpus-per-task=2", "--mem=16G", f"--time={args.finalize_walltime}",
        f"--exclude={args.exclude}",
    ]
    if dependencies:
        finalizer_command.append(f"--dependency=afterok:{dependencies}")
    finalizer_command.extend([f"--export={finalizer_export}", str(FINALIZER)])
    finalizer_job_id = submit(finalizer_command, args.dry_run)
    payload["finalization_job"] = {
        "job_id": finalizer_job_id,
        "environment": finalizer_env,
        "command": finalizer_command,
        "dependency": f"afterok:{dependencies}" if dependencies else None,
    }
    rewrite_ledger(ledger_path, payload)
    print(f"submitted six Qwen3-14B trainings, base {base_job_id}, and finalizer {finalizer_job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("canary", "full"), required=True)
    parser.add_argument("--canary-job-id", default="")
    parser.add_argument("--canary-ledger", type=Path)
    parser.add_argument("--account", default="allcs")
    parser.add_argument("--partition", default="cs")
    parser.add_argument("--exclude", default="node206")
    parser.add_argument("--canary-walltime", default="08:00:00")
    parser.add_argument("--walltime", default="30:00:00")
    parser.add_argument("--base-walltime", default="06:00:00")
    parser.add_argument("--finalize-walltime", default="01:00:00")
    parser.add_argument("--advance-walltime", default="00:30:00")
    parser.add_argument("--vllm-ratio", type=float, default=0.78)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.70 <= args.vllm_ratio <= 0.85:
        parser.error("--vllm-ratio must be in [0.70, 0.85]")
    if args.stage == "full" and (not args.canary_job_id or args.canary_ledger is None):
        parser.error("--stage full requires --canary-job-id and --canary-ledger")
    if args.stage == "canary":
        launch_canary(args)
    else:
        launch_full(args)


if __name__ == "__main__":
    main()
