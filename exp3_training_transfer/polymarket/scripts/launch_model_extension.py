#!/usr/bin/env python3
"""Launch Qwen3-8B/Llama-3.1-8B on the unchanged locked Exp3B protocol.

Real submission is impossible until the six synthetic C3 mechanism canaries pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data" / "exp3b_registered"
TRAIN = ROOT / "scripts" / "polymarket_rl_model_extension.sh"
LOCKED = ROOT / "scripts" / "locked_test_model_extension.sbatch"
EVALUATOR = ROOT / "scripts" / "evaluate_locked_test_model_extension.py"
TRAINER = REPO / "exp3_training_transfer" / "mechanism_family" / "run_mechanism_rl.py"
CANARY_AUDIT = REPO / "exp3_training_transfer" / "mechanism_family" / "audit_canaries.py"
CANARY_REPORT = REPO / "exp3_training_transfer" / "mechanism_family" / "protocol" / "canary_audit.json"
PREFLIGHT_REPORT = ROOT / "protocol" / "exp3b_preflight.json"
BASELINE_REPORT = ROOT / "reports" / "exp3b_baselines.json"
RUNS = ROOT / "runs"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"
MODELS = {
    "qwen3_8b": {"id": "Qwen/Qwen3-8B", "template": "auto_no_think"},
    "llama3_1_8b": {"id": "meta-llama/Llama-3.1-8B-Instruct", "template": "auto"},
}
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_inputs() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    preflight = json.loads(PREFLIGHT_REPORT.read_text())
    baseline = json.loads(BASELINE_REPORT.read_text())
    if manifest.get("protocol_version") != "exp3b_registered_v1":
        raise SystemExit("frozen dataset protocol changed")
    if preflight.get("status") != "pass" or preflight.get("counts") != {
            "train": 1736, "dev": 512, "test": 1024}:
        raise SystemExit("frozen preflight is not a registered pass")
    if baseline.get("protocol_version") != "exp3b_registered_v1":
        raise SystemExit("frozen baseline protocol changed")
    for split in ("train", "dev", "test"):
        path = DATA / f"{split}.tasks.jsonl"
        if sha256(path) != manifest["output_sha256"][path.name]:
            raise SystemExit(f"frozen {split} split hash changed")


def cache_path(model_id: str) -> Path:
    return HF_HUB / ("models--" + model_id.replace("/", "--"))


def submit(command: list[str], dry_run: bool) -> str | None:
    print(shlex.join(command))
    if dry_run:
        return None
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    print(f"submitted job {job_id}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--partition", default="all")
    parser.add_argument("--exclude", default="node206")
    parser.add_argument("--walltime", default="20:00:00")
    parser.add_argument("--collocate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.seeds)) != list(SEEDS):
        parser.error("the extension requires exactly seeds 42 43 44")

    validate_frozen_inputs()
    missing = [MODELS[key]["id"] for key in args.models
               if not cache_path(MODELS[key]["id"]).is_dir()]
    if missing:
        raise SystemExit("offline model cache missing: " + ", ".join(missing))
    if not args.dry_run:
        gate = subprocess.run([sys.executable, str(CANARY_AUDIT), "--write", str(CANARY_REPORT)],
                              cwd=REPO, text=True, capture_output=True, check=False)
        if gate.returncode:
            print(gate.stdout, end="")
            print(gate.stderr, end="", file=sys.stderr)
            raise SystemExit("synthetic canary gate failed; Polymarket extension remains locked")
        print(gate.stdout.strip())

    submissions, evaluations = [], []
    gpus = 1 if args.collocate else 2
    for model_key in args.models:
        model = MODELS[model_key]
        model_jobs = []
        for seed in sorted(args.seeds):
            env = {"MODEL_KEY": model_key, "MODEL": model["id"],
                   "PROMPT_TEMPLATE": model["template"], "SEED": str(seed),
                   "COLLOCATE": "1" if args.collocate else "0", "GPUS": str(gpus),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
            command = ["sbatch", "--parsable", f"--partition={args.partition}",
                       f"--gres=gpu:a6000:{gpus}", f"--time={args.walltime}",
                       f"--exclude={args.exclude}", f"--export={export}", str(TRAIN)]
            job_id = submit(command, args.dry_run)
            model_jobs.append((seed, job_id))
            submissions.append({"model": model_key, "seed": seed, "job_id": job_id,
                                "environment": env, "command": command})

        adapter_spec = ";".join(f"{seed}:{job or 'DRYRUN'}" for seed, job in model_jobs)
        dependency = ":".join(job for _, job in model_jobs if job)
        env = {"MODEL_KEY": model_key, "MODEL": model["id"],
               "PROMPT_TEMPLATE": model["template"], "TRAIN_ADAPTERS": adapter_spec}
        export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
        command = ["sbatch", "--parsable", f"--partition={args.partition}",
                   "--gres=gpu:a6000:1", "--time=03:00:00", f"--exclude={args.exclude}"]
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend([f"--export={export}", str(LOCKED)])
        job_id = submit(command, args.dry_run)
        evaluations.append({"model": model_key, "job_id": job_id,
                            "adapter_spec": adapter_spec, "command": command})

    print(f"rendered {len(submissions)} training and {len(evaluations)} locked-eval jobs")
    if args.dry_run:
        print("dry run: nothing submitted")
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = {
        "protocol_version": "exp3b_model_extension_v1",
        "frozen_parent_protocol": "exp3b_registered_v1", "submitted_at": timestamp,
        "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
        "preflight_sha256": sha256(PREFLIGHT_REPORT),
        "baseline_sha256": sha256(BASELINE_REPORT),
        "synthetic_canary_audit_sha256": sha256(CANARY_REPORT),
        "train_script_sha256": sha256(TRAIN), "locked_script_sha256": sha256(LOCKED),
        "trainer_sha256": sha256(TRAINER), "evaluator_sha256": sha256(EVALUATOR),
        "submissions": submissions, "locked_evaluations": evaluations,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"exp3b_model_extension_{timestamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"ledger -> {path}")


if __name__ == "__main__":
    main()
