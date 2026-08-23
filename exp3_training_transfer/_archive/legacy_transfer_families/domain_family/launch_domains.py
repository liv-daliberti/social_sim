#!/usr/bin/env python3
"""Registered submission for the Experiment-3 domain-transfer campaign (exp3d).

Refuses to submit unless the preflight passes, and records exactly what was launched -- git
revision, preflight manifest hash, trainer and launcher hashes, and the full sbatch command and
environment for every job -- so a result can always be traced back to the bytes that produced it.

    python launch_domains.py --canary                 # one job (d3 seed 42), to prove the pipeline
    python launch_domains.py --all                    # 4 arms x 3 seeds = 12 jobs
    python launch_domains.py --arms d1 d2 --seeds 42  # a subset
    python launch_domains.py --all --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ARMS = ("d1", "d2", "d3", "structureless")
SEEDS = (42, 43, 44)
MANIFEST = ROOT / "protocol" / "exp3d_manifest.json"
TRAINER = REPO / "exp3_training_transfer" / "biased_news" / "run_biased_news_rl.py"
LAUNCHER = ROOT / "domain_rl.sh"

BASE_ENV = {
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _preflight() -> None:
    """Gate: never book a GPU on data that has not just passed every check."""
    proc = subprocess.run([sys.executable, str(ROOT / "preflight_domains.py"),
                           "--write-manifest", str(MANIFEST)],
                          cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout + proc.stderr, file=sys.stderr)
        raise SystemExit("preflight FAILED -- nothing submitted")
    print(proc.stdout.strip())


def submit(arm: str, seed: int, dry_run: bool, walltime: str = "12:00:00",
           collocate: bool = True, exclude: str | None = None) -> dict:
    # Without collocation the actor and learner need a GPU each, so the sbatch --gres request must
    # match what domain_rl.sh will ask oat for, or the job books one GPU and stalls on the second.
    gpus = 1 if collocate else 2
    env = dict(BASE_ENV, ARM=arm, TAG=arm, SEED=str(seed),
               COLLOCATE="1" if collocate else "0", GPUS=str(gpus))
    exports = "ALL," + ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", "--parsable", "--partition=all", f"--gres=gpu:a6000:{gpus}",
           f"--time={walltime}"]
    if exclude:
        # 2026-08-12: node206 handed four concurrent jobs the SAME physical GPU, so each OOM'd
        # during model load ("37 MiB free", siblings holding 11.3/11.2/4.5 GiB). Exclude nodes
        # whose GPU isolation cannot be trusted rather than re-rolling the dice on placement.
        cmd.append(f"--exclude={exclude}")
    cmd += [f"--export={exports}", str(LAUNCHER)]
    if dry_run:
        job_id = "DRYRUN"
    else:
        job_id = subprocess.run(cmd, check=True, capture_output=True,
                                text=True).stdout.strip().split(";")[0]
    print(f"  {arm:<14} seed {seed}  -> job {job_id}")
    return {"arm": arm, "seed": seed, "job_id": job_id, "command": cmd, "environment": env}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--canary", action="store_true",
                    help="submit only d3/seed42 to prove the pipeline trains before booking the rest")
    ap.add_argument("--time", default="12:00:00",
                    help="sbatch wall limit. A canary only needs to reach a training step, and a "
                         "short request is far easier for the backfill scheduler to place than the "
                         "full 8h reservation -- use e.g. 1:00:00 to prove the pipeline quickly.")
    ap.add_argument("--no-collocate", action="store_true",
                    help="give the actor and learner a GPU each (2 GPUs/job) instead of sharing one. "
                         "Avoids the shared-GPU init deadlock seen since the 2026-07-14 node rebuild.")
    ap.add_argument("--exclude", help="comma-separated nodes to avoid (e.g. node206)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.canary:
        arms, seeds = ["d3"], [42]
    elif args.all:
        arms, seeds = list(ARMS), list(SEEDS)
    else:
        arms = args.arms or list(ARMS)
        seeds = args.seeds or list(SEEDS)

    _preflight()
    print(f"\nsubmitting {len(arms) * len(seeds)} job(s):")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submissions = [submit(a, s, args.dry_run, args.time, not args.no_collocate, args.exclude)
                   for a in arms for s in seeds]

    ledger = {
        "protocol": "exp3d_domain_transfer_v1",
        "submitted_at": stamp,
        "git_revision": _git_revision(),
        "preflight_manifest": str(MANIFEST.relative_to(REPO)),
        "preflight_manifest_sha256": _sha256(MANIFEST),
        "trainer_sha256": _sha256(TRAINER),
        "launcher_sha256": _sha256(LAUNCHER),
        "walltime": args.time,
        "collocate": not args.no_collocate,
        "submissions": submissions,
    }
    if not args.dry_run:
        out = ROOT / "runs" / f"exp3d_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nledger -> {out}")


if __name__ == "__main__":
    main()
