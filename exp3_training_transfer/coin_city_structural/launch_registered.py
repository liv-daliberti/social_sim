#!/usr/bin/env python3
"""Register and submit the gated Coin City A-to-B model roster."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = ROOT / "runs"
MANIFEST = ROOT / "protocol" / "coin_city_structural_manifest.json"
MODELS = {
    "qwen3_4b": ("Qwen/Qwen3-4B-Instruct-2507", "biased_news"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "auto_no_think"),
    "llama3_1_8b": ("meta-llama/Llama-3.1-8B-Instruct", "auto"),
}
ARMS = ("causal", "population_prior")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(command: list[str], dry_run: bool) -> str | None:
    print(shlex.join(command))
    if dry_run:
        return None
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def sbatch(script: Path, env: dict[str, str], *, gpus: int, memory: str, walltime: str,
           dependency: str | None, partition: str) -> list[str]:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = ["sbatch", "--parsable", f"--partition={partition}", "--cpus-per-task=8",
               f"--mem={memory}", f"--time={walltime}", f"--export={export}"]
    if gpus:
        command.append(f"--gres=gpu:a6000:{gpus}")
        command.append("--exclude=node206")
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(script))
    return command


def train_env(model_key: str, arm: str, seed: int, *, canary: bool) -> dict[str, str]:
    model, template = MODELS[model_key]
    return {
        "ARM": arm, "MODEL_KEY": model_key, "MODEL": model, "PROMPT_TEMPLATE": template,
        "SEED": str(seed), "TAG": f"{'canary_' if canary else ''}{arm}_{model_key}_s{seed}",
        "MAX_TRAIN": "160" if canary else "4800",
        "EVAL_STEPS": "10" if canary else "50",
        "GPUS": "2", "BATCH": "16", "ROLLOUT_PER_PROMPT": "8",
        "LORA_RANK": "32", "LORA_ALPHA": "64", "VLLM_RATIO": "0.38",
        "MAX_MODEL_LEN": "3072", "PROMPT_MAX_LENGTH": "2304", "GEN_LEN": "192",
        "LR": "0.000001", "TEMP": "1.3", "RESPONSE_W": "0.60",
        "EVAL_BATCH": "120", "ZERO_STAGE": "2", "STOCHASTIC_N": "5",
        "STOCHASTIC_TEMP": "0.7", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-job-ids", nargs="+", required=True,
                        help="current campaign jobs; canaries wait for afterany on every ID")
    parser.add_argument("--partition", default="cs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not value.isdigit() for value in args.after_job_ids):
        parser.error("every dependency must be a numeric Slurm job ID")

    subprocess.run([sys.executable, str(ROOT / "preflight.py")], cwd=ROOT, check=True)
    preflight = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise SystemExit("preflight is not passed")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUNS.mkdir(parents=True, exist_ok=True)
    ledger_path = RUNS / f"coin_city_structural_{timestamp}.json"
    barrier = "afterany:" + ":".join(dict.fromkeys(args.after_job_ids))
    ledger = {
        "protocol": "coin_city_structural_transfer_v1",
        "submitted_at": timestamp,
        "upstream_dependency": {"semantics": "afterany", "job_ids": args.after_job_ids},
        "preflight_manifest": str(MANIFEST.relative_to(REPO)),
        "preflight_sha256": sha256(MANIFEST),
        "structured_decoding": {"kind": "gbnf", "backend": "xgrammar",
                                "contract": "one ten-number forecasts array",
                                "value_constraints": "none"},
        "source_sha256": {path.name: sha256(path) for path in (
            ROOT / "worlds.py", ROOT / "prompt.py", ROOT / "make_dataset.py",
            ROOT / "preflight.py", ROOT / "report.py", ROOT / "train.sh",
            ROOT / "base_eval.sh", ROOT / "canary_audit.py",
        )},
        "canaries": [], "full_training": [], "base_evaluations": [],
    }
    for model_key in MODELS:
        for arm in ARMS:
            env = train_env(model_key, arm, 42, canary=True)
            command = sbatch(ROOT / "train.sh", env, gpus=2, memory="100G",
                             walltime="08:00:00", dependency=barrier, partition=args.partition)
            job_id = submit(command, args.dry_run)
            ledger["canaries"].append({"model": model_key, "arm": arm, "seed": 42,
                                       "job_id": job_id, "environment": env, "command": command})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    canary_ids = [row["job_id"] for row in ledger["canaries"] if row["job_id"]]
    audit_env = {"CAMPAIGN_LEDGER": str(ledger_path)}
    audit_command = sbatch(
        ROOT / "canary_audit.sbatch", audit_env, gpus=0, memory="8G", walltime="00:20:00",
        dependency=("afterok:" + ":".join(canary_ids)) if canary_ids else None,
        partition=args.partition,
    )
    audit_job = submit(audit_command, args.dry_run)
    ledger["canary_gate"] = {"job_id": audit_job, "command": audit_command,
                             "environment": audit_env, "semantics": "afterok"}
    full_dependency = f"afterok:{audit_job}" if audit_job else None

    for model_key in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                env = train_env(model_key, arm, seed, canary=False)
                command = sbatch(ROOT / "train.sh", env, gpus=2, memory="100G",
                                 walltime="1-00:00:00", dependency=full_dependency,
                                 partition=args.partition)
                job_id = submit(command, args.dry_run)
                ledger["full_training"].append({
                    "kind": "confirmatory", "model": model_key, "arm": arm, "seed": seed,
                    "job_id": job_id, "environment": env, "command": command,
                })
    for seed in SEEDS:
        env = train_env("qwen3_4b", "structureless", seed, canary=False)
        command = sbatch(ROOT / "train.sh", env, gpus=2, memory="100G",
                         walltime="1-00:00:00", dependency=full_dependency,
                         partition=args.partition)
        job_id = submit(command, args.dry_run)
        ledger["full_training"].append({
            "kind": "diagnostic", "model": "qwen3_4b", "arm": "structureless",
            "seed": seed, "job_id": job_id, "environment": env, "command": command,
        })
    for model_key, (model, template) in MODELS.items():
        env = {"MODEL_KEY": model_key, "MODEL": model, "PROMPT_TEMPLATE": template}
        command = sbatch(ROOT / "base_eval.sh", env, gpus=1, memory="24G",
                         walltime="02:00:00", dependency=full_dependency,
                         partition=args.partition)
        job_id = submit(command, args.dry_run)
        ledger["base_evaluations"].append({
            "model": model_key, "job_id": job_id, "environment": env, "command": command,
        })
    ledger["counts"] = {"canary_training": 6, "gate": 1, "confirmatory_training": 18,
                        "diagnostic_training": 3, "base_evaluations": 3,
                        "post_gate_scientific_jobs": 24}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ledger -> {ledger_path}")
    print("submitted 6 gated canaries + 1 audit + 18 confirmatory + 3 diagnostic + 3 base jobs")


if __name__ == "__main__":
    main()
