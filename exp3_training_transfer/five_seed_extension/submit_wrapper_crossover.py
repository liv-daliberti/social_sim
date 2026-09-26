#!/usr/bin/env python3
"""Re-run Qwen3-8B seeds 42-44 under the current train.sh, as a wrapper control.

train.sh was modified on 2026-08-28, after Coin City seeds 42--44 trained and
before seeds 45--49. Two flags changed and both apply silently because
`coin_env` sets neither:

    --save_steps 999999  ->  --save_steps 40      (SAVE_STEPS default)
    (absent)             ->  --save_ckpt          (SAVE_CKPT default)

So the eight-seed roster mixes two wrappers, and the seeds 42-44 versus 45-49
split-half is confounded with wrapper version. This script re-runs seeds 42, 43
and 44 in both arms under the *current* wrapper, holding the seed fixed, so the
wrapper effect can be read directly: compare each seed's contrast here against
its archived value from the original run.

These runs are a diagnostic control. They are NOT endpoints, are NOT part of
any estimand, and must never enter the eight-seed roster. They are tagged
`wrapxover_` so their report directories cannot match the renderer's
`{arm}_{model}_s{seed}_*` glob.

Dry run by default; pass --submit to submit.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import COIN, REPO, RUNS, coin_env, gpu_command  # noqa: E402

PROTOCOL = "exp3_coin_city_wrapper_crossover_v1"
MODEL_KEY = "qwen3_8b"
SEEDS = (42, 43, 44)
ARMS = ("causal", "population_prior")
TAG_PREFIX = "wrapxover_"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records() -> list[dict]:
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            environment = coin_env(MODEL_KEY, arm, seed)
            # Only the output directory name changes; TAG feeds nothing but
            # $OUT in train.sh, and the scored identity fields come from
            # MODEL_KEY/ARM/SEED, so these rows remain directly comparable to
            # the originals.
            environment = environment | {
                "TAG": f"{TAG_PREFIX}{arm}_{MODEL_KEY}_s{seed}"}
            command = gpu_command(
                COIN / "train.sh", environment,
                gpus=2, cpus=8, memory="100G", walltime="30:00:00",
                partition="cs,all", account="allcs", qos="medium",
            )
            out.append({"arm": arm, "seed": seed, "environment": environment,
                        "command": command})
    return out


def guard() -> None:
    existing = sorted((COIN / "reports").glob(f"{TAG_PREFIX}*_{MODEL_KEY}_s*"))
    if existing:
        raise SystemExit("crossover report directories already exist:\n  "
                         + "\n  ".join(p.name for p in existing))
    for arm in ARMS:
        for seed in SEEDS:
            original = sorted((COIN / "reports").glob(f"{arm}_{MODEL_KEY}_s{seed}_*"))
            if len(original) != 1:
                raise SystemExit(
                    f"{arm} s{seed}: expected exactly one original report root to "
                    f"compare against, found {[p.name for p in original]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    guard()
    roster = records()
    for record in roster:
        print(shlex.join(record["command"]), flush=True)
    print(f"\n{len(roster)} crossover job(s); submit={args.submit}", flush=True)
    if not args.submit:
        return

    for record in roster:
        output = subprocess.check_output(record["command"], cwd=REPO, text=True).strip()
        job_id = output.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch output: {output!r}")
        record["job_id"] = job_id
        print(f"submitted {job_id}  {record['arm']} s{record['seed']}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = {
        "created_at": stamp,
        "kind": "coin_city_wrapper_crossover",
        "protocol": PROTOCOL,
        "purpose": "Measure whether the 2026-08-28 train.sh change (--save_steps "
                   "999999 -> 40, --save_ckpt added) affects the trained endpoint, "
                   "by re-running seeds 42-44 under the current wrapper and "
                   "comparing each seed's contrast to its archived original.",
        "model_key": MODEL_KEY,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "tag_prefix": TAG_PREFIX,
        "is_endpoint": False,
        "enters_any_estimand": False,
        "wrapper_under_test": {
            "old": {"save_steps": "999999", "save_ckpt": False},
            "new": {"save_steps": "40", "save_ckpt": True},
        },
        "training_jobs": roster,
        "code_sha256": {
            "exp3_training_transfer/coin_city_structural/train.sh":
                sha256(REPO / "exp3_training_transfer/coin_city_structural/train.sh"),
            "exp3_training_transfer/five_seed_extension/submit_wrapper_crossover.py":
                sha256(Path(__file__)),
        },
    }
    path = RUNS / f"coin_city_wrapper_crossover_{stamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"\nledger -> {path}", flush=True)


if __name__ == "__main__":
    main()
