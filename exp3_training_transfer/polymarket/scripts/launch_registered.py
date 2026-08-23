#!/usr/bin/env python3
"""Submit the registered Experiment 3B seed triplet after all audits pass."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data" / "exp3b_registered"
PREFLIGHT = ROOT / "scripts" / "preflight_exp3b.py"
BASELINES = ROOT / "scripts" / "evaluate_baselines.py"
SLURM = ROOT / "scripts" / "polymarket_rl.sh"
LOCKED_TEST_SLURM = ROOT / "scripts" / "locked_test.sbatch"
PREFLIGHT_REPORT = ROOT / "protocol" / "exp3b_preflight.json"
BASELINE_REPORT = ROOT / "reports" / "exp3b_baselines.json"
RUNS = ROOT / "runs"
FROZEN_ENV = {
    "DATA": str(DATA),
    "TAG": "market",
    "MODEL": "Qwen/Qwen3-4B-Instruct-2507",
    "VLLM_RATIO": "0.40",
    "LORA_RANK": "32",
    "LORA_ALPHA": "64",
    "BATCH": "16",
    "ROLLOUT_PER_PROMPT": "8",
    "MAX_TRAIN": "4800",
    "TEMP": "1.3",
    "LR": "0.000001",
    "EVAL_STEPS": "25",
    "GEN_LEN": "128",
    "MAX_MODEL_LEN": "1920",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_reports() -> None:
    required_inputs = (
        DATA / "manifest.json",
        DATA / "train.tasks.jsonl",
        DATA / "dev.tasks.jsonl",
        DATA / "test.tasks.jsonl",
    )
    for report, dependencies in (
        (PREFLIGHT_REPORT, required_inputs + (PREFLIGHT,)),
        (BASELINE_REPORT, required_inputs + (BASELINES,)),
    ):
        if not report.exists():
            raise SystemExit(f"missing frozen audit report: {report}; no jobs submitted")
        if report.stat().st_mtime < max(path.stat().st_mtime for path in dependencies):
            raise SystemExit(f"stale frozen audit report: {report}; no jobs submitted")
    preflight = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass" or preflight.get("counts") != {
        "train": 1736,
        "dev": 512,
        "test": 1024,
    }:
        raise SystemExit("frozen preflight report is not a registered pass; no jobs submitted")
    baselines = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    if baselines.get("protocol_version") != "exp3b_registered_v1":
        raise SystemExit("frozen baseline report has the wrong protocol; no jobs submitted")
    print("frozen audit reports PASS")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--partition", default="all")
    parser.add_argument("--walltime", default="08:00:00")
    parser.add_argument(
        "--afterok",
        help="gate every training job on a successful Slurm job (for a GPU smoke test)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="submit only after separately generated frozen audit reports pass validation",
    )
    args = parser.parse_args()
    if sorted(set(args.seeds)) != [42, 43, 44]:
        parser.error("the registered protocol requires exactly seeds 42 43 44")

    if not args.submit_only:
        for command, label in (
            ([sys.executable, str(PREFLIGHT), "--data", str(DATA), "--write", str(PREFLIGHT_REPORT)], "preflight"),
            ([sys.executable, str(BASELINES), "--data", str(DATA), "--output", str(BASELINE_REPORT)], "baselines"),
        ):
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
            if result.returncode:
                print(result.stdout, end="")
                print(result.stderr, end="", file=sys.stderr)
                raise SystemExit(f"Experiment 3B {label} failed; no jobs submitted")
            print(f"{label} PASS")
    validate_frozen_reports()

    submissions = []
    for seed in sorted(args.seeds):
        environment = dict(FROZEN_ENV)
        environment["SEED"] = str(seed)
        export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in environment.items())
        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partition}",
            "--gres=gpu:a6000:1",
            f"--time={args.walltime}",
            f"--export={export_arg}",
            str(SLURM),
        ]
        if args.afterok:
            command.insert(2, f"--dependency=afterok:{args.afterok}")
        print(shlex.join(command))
        job_id = None
        if not args.dry_run:
            job_id = subprocess.check_output(command, cwd=REPO, text=True).strip()
            print(f"submitted Experiment 3B seed {seed}: job {job_id}")
        submissions.append({"seed": seed, "job_id": job_id, "command": command, "environment": environment})

    if args.dry_run:
        print("dry run: no jobs submitted")
        return
    adapter_spec = ";".join(
        f"{item['seed']}:{str(item['job_id']).split(';', 1)[0]}"
        for item in submissions
    )
    dependency = ":".join(
        str(item["job_id"]).split(";", 1)[0] for item in submissions
    )
    locked_test_command = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        "--gres=gpu:a6000:1",
        "--time=02:00:00",
        f"--dependency=afterok:{dependency}",
        f"--export=ALL,TRAIN_ADAPTERS={adapter_spec}",
        str(LOCKED_TEST_SLURM),
    ]
    print(shlex.join(locked_test_command))
    locked_test_job_id = subprocess.check_output(
        locked_test_command, cwd=REPO, text=True
    ).strip()
    print(f"submitted locked final evaluation: job {locked_test_job_id}")

    RUNS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = {
        "protocol_version": "exp3b_registered_v1",
        "submitted_at": timestamp,
        "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
        "preflight_sha256": sha256(PREFLIGHT_REPORT),
        "baseline_report_sha256": sha256(BASELINE_REPORT),
        "slurm_script_sha256": sha256(SLURM),
        "locked_test_slurm_sha256": sha256(LOCKED_TEST_SLURM),
        "trainer_sha256": sha256(REPO / "exp3_training_transfer" / "biased_news" / "run_biased_news_rl.py"),
        "scorer_sha256": sha256(REPO / "exp3_training_transfer" / "biased_news" / "forecast_scoring.py"),
        "training_evaluation_split": "dev",
        "locked_final_split": "test",
        "submissions": submissions,
        "locked_test_submission": {
            "job_id": locked_test_job_id,
            "command": locked_test_command,
            "adapter_spec": adapter_spec,
        },
    }
    ledger_path = RUNS / f"exp3b_{timestamp}.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"submission ledger: {ledger_path}")


if __name__ == "__main__":
    main()
