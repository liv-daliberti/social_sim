#!/usr/bin/env python3
"""Submit the frozen Experiment 3A protocol and record exact Slurm job IDs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SLURM_SCRIPT = ROOT / "dag_rl.sh"
PREFLIGHT = ROOT / "preflight_exp3a.py"
MANIFEST = ROOT / "protocol" / "exp3a_manifest.json"
RUNS = ROOT / "runs"
ARMS = {
    "family": (ROOT / "data" / "rl_multishock", "fam"),
    "structureless": (ROOT / "data" / "rl_multishock_control", "ctrl"),
    "prior_mean": (ROOT / "data" / "rl_multishock_priormean", "pmean"),
}
COMPLETED_ARMS = {"family", "structureless"}
FROZEN_ENV = {
    "MODEL": "Qwen/Qwen3-4B-Instruct-2507",
    "VLLM_RATIO": "0.40",
    "LORA_RANK": "32",
    "LORA_ALPHA": "64",
    "BATCH": "16",
    "MAX_TRAIN": "4800",
    "TEMP": "1.3",
    "REWARD_SCALE": "15",
    "SLOPE_W": "0.5",
    "SLOPE_SCALE": "0.5",
    "LR": "0.000001",
    "EVAL_STEPS": "25",
    "GEN_LEN": "1024",
    "MAX_MODEL_LEN": "2560",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARMS), default="prior_mean")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--partition", default="all")
    parser.add_argument("--walltime", default="08:00:00")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Allow resubmitting the already-completed family/control arms.",
    )
    args = parser.parse_args()

    if sorted(set(args.seeds)) != [42, 43, 44]:
        parser.error("the registered protocol requires exactly seeds 42 43 44")
    if args.arm in COMPLETED_ARMS and not args.allow_repeat:
        parser.error(
            f"{args.arm} already has complete 300-step runs; pass --allow-repeat "
            "only for an intentional replication"
        )

    audit = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--write-manifest", str(MANIFEST)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if audit.returncode:
        print(audit.stdout, end="")
        print(audit.stderr, end="", file=sys.stderr)
        raise SystemExit("Experiment 3A preflight failed; no jobs submitted")
    print(f"preflight PASS: {MANIFEST}")

    data, tag = ARMS[args.arm]
    submissions = []
    for seed in sorted(args.seeds):
        exported = dict(FROZEN_ENV)
        exported.update({"DATA": str(data), "TAG": tag, "SEED": str(seed)})
        export_arg = "ALL," + ",".join(
            f"{key}={value}" for key, value in exported.items()
        )
        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partition}",
            "--gres=gpu:a6000:1",
            f"--time={args.walltime}",
            f"--export={export_arg}",
            str(SLURM_SCRIPT),
        ]
        print(shlex.join(command))
        job_id = None
        if not args.dry_run:
            job_id = subprocess.check_output(command, cwd=REPO, text=True).strip()
            print(f"submitted {args.arm} seed {seed}: job {job_id}")
        submissions.append(
            {
                "arm": args.arm,
                "seed": seed,
                "job_id": job_id,
                "command": command,
                "environment": exported,
            }
        )

    if args.dry_run:
        print("dry run: no jobs submitted")
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = {
        "protocol": "exp3a_registered_v1",
        "submitted_at": timestamp,
        "git_revision": _git_revision(),
        "preflight_manifest": str(MANIFEST.relative_to(REPO)),
        "preflight_manifest_sha256": _sha256(MANIFEST),
        "slurm_script_sha256": _sha256(SLURM_SCRIPT),
        "trainer_sha256": _sha256(
            REPO / "exp3_training_transfer" / "biased_news" / "run_biased_news_rl.py"
        ),
        "submissions": submissions,
    }
    ledger_path = RUNS / f"exp3a_{args.arm}_{timestamp}.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"submission ledger: {ledger_path}")


if __name__ == "__main__":
    main()
