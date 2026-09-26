#!/usr/bin/env python3
"""Submit the Qwen3-8B Coin City added seeds 45-49 under the eight-seed amendment.

Implements exactly the roster in QWEN3_8B_EIGHT_SEED_AMENDMENT.md
(`exp3_qwen3_8b_coin_city_eight_seed_v1`): five added seeds by two confirmatory
arms, ten training jobs, A6000 via allcs on cs,all, matching the seed-42--44
GPU architecture.

The per-job environment and the sbatch allocation come from ``launch.py``'s own
``coin_env`` and ``gpu_command``, so what is submitted here is the frozen parent
protocol code with a different seed tuple rather than a retyped copy of it.

Dry run by default; pass --submit to actually submit.

    python exp3_training_transfer/five_seed_extension/submit_qwen3_8b_eight_seed.py
    python ... --submit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import pathlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import COIN, REPO, RUNS, coin_env, gpu_command  # noqa: E402

AMENDMENT = "exp3_qwen3_8b_coin_city_eight_seed_v1"
MODEL_KEY = "qwen3_8b"
ADDED_SEEDS = (45, 46, 47, 48, 49)
ARMS = ("causal", "population_prior")

# Bound at amendment time; re-verified on every run so a drifted wrapper or a
# reworded protocol stops the submission instead of silently changing it.
PINNED = {
    "exp3_training_transfer/five_seed_extension/PROTOCOL.md":
        "4c485b67a497fb36a3b866a964090af067e761bcf9512aca7e9723ebcf9d89ed",
    "exp3_training_transfer/five_seed_extension/QWEN3_8B_EIGHT_SEED_AMENDMENT.md":
        "5d12d8a64922e73f5d63e55cdff8421d19cd54a59075567e2f8d5d5221810179",
    "exp3_training_transfer/coin_city_structural/train.sh":
        "7ca287c389ff4f18678de9e381e5de6b0064cdd2635eebbfab3a749476685a21",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins() -> None:
    for rel, expected in PINNED.items():
        actual = sha256(REPO / rel)
        if actual != expected:
            raise SystemExit(
                f"hash drift, refusing to submit:\n  {rel}\n"
                f"  expected {expected}\n  actual   {actual}"
            )


# One complete endpoint is 1,440 greedy rows and 1,440 x 5 stochastic rows.
GREEDY_ROWS = 1440
STOCHASTIC_ROWS = 1440 * 5


def cell_state(arm: str, seed: int) -> tuple[str, list[pathlib.Path]]:
    """'complete', 'partial' or 'absent', with the report directories found.

    A cell is complete only if exactly one directory holds both score files at
    their full registered row counts. Anything else is a failed attempt whose
    directory must be moved aside before the cell is retried, so the renderer
    never sees two candidate endpoints for one cell.
    """
    dirs = sorted((COIN / "reports").glob(f"{arm}_{MODEL_KEY}_s{seed}_*"))
    complete = []
    for d in dirs:
        greedy, stochastic = d / "greedy.scores.jsonl", d / "stochastic_n5.scores.jsonl"
        if (greedy.is_file() and stochastic.is_file()
                and sum(1 for _ in greedy.open()) == GREEDY_ROWS
                and sum(1 for _ in stochastic.open()) == STOCHASTIC_ROWS):
            complete.append(d)
    if len(complete) == 1 and len(dirs) == 1:
        return "complete", dirs
    if complete:
        return "ambiguous", dirs
    return ("partial" if dirs else "absent"), dirs


def partition_roster(roster: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop cells that already hold a complete endpoint; never submit one twice."""
    submit, skipped = [], []
    for record in roster:
        state, dirs = cell_state(record["arm"], record["seed"])
        label = f"{record['arm']} s{record['seed']}"
        if state == "complete":
            skipped.append(f"{label}: complete, skipping ({dirs[0].name})")
        elif state == "ambiguous":
            raise SystemExit(
                f"{label}: a complete endpoint coexists with other attempt "
                f"directories; move the failed ones under reports/failed_attempts/ "
                f"before continuing:\n  " + "\n  ".join(d.name for d in dirs))
        else:
            if dirs:
                raise SystemExit(
                    f"{label}: incomplete attempt directory still in place; move it "
                    f"under reports/failed_attempts/ before retrying:\n  "
                    + "\n  ".join(d.name for d in dirs))
            submit.append(record)
    return submit, skipped


def verify_inputs() -> None:
    for arm in ARMS:
        for split in ("train", "heldout"):
            path = COIN / "data" / arm / split
            if not path.is_dir():
                raise SystemExit(f"missing dataset: {path}")
    snapshot = (REPO / ".runtime/hf_home/hub/models--Qwen--Qwen3-8B/snapshots"
                / "b968826d9c46dd6066d109eabc6255188de91218")
    if not snapshot.is_dir():
        raise SystemExit(f"missing pinned offline snapshot: {snapshot}")


def records() -> list[dict]:
    out = []
    for arm in ARMS:
        for seed in ADDED_SEEDS:
            environment = coin_env(MODEL_KEY, arm, seed)
            command = gpu_command(
                COIN / "train.sh", environment,
                gpus=2, cpus=8, memory="100G", walltime="30:00:00",
                partition="cs,all", account="allcs", qos="medium",
            )
            out.append({"arm": arm, "seed": seed, "identity": f"{MODEL_KEY}/{arm}",
                        "environment": environment, "command": command})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true",
                        help="actually submit; otherwise print the commands only")
    parser.add_argument("--only", type=int, nargs="+", metavar="SEED",
                        help="restrict to these added seeds (infrastructural "
                             "retry only; never for result inspection)")
    parser.add_argument("--arm", choices=ARMS, action="append", dest="arms",
                        help="restrict to these arms; repeatable")
    args = parser.parse_args()

    verify_pins()
    verify_inputs()

    roster = records()
    if args.only:
        roster = [r for r in roster if r["seed"] in set(args.only)]
    if args.arms:
        roster = [r for r in roster if r["arm"] in set(args.arms)]
    if not roster:
        raise SystemExit("seed/arm filters matched no roster entry")

    roster, skipped = partition_roster(roster)
    for line in skipped:
        print(f"skip  {line}", flush=True)
    if not roster:
        print("\nnothing to submit; every selected cell already has a complete endpoint")
        return

    for record in roster:
        print(shlex.join(record["command"]), flush=True)
    print(f"\n{len(roster)} job(s); submit={args.submit}", flush=True)
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
        "kind": "qwen3_8b_eight_seed_submission",
        "protocol": AMENDMENT,
        "parent_protocol": "exp3_exp4_five_seed_extension_v1",
        "model_key": MODEL_KEY,
        "added_seeds": list(ADDED_SEEDS),
        "combined_seed_roster": [42, 43, 44, *ADDED_SEEDS],
        "arms": list(ARMS),
        "submitted": True,
        "scientific_hyperparameters_changed": False,
        "training_jobs": [
            {k: v for k, v in r.items() if k != "command"} | {"command": r["command"]}
            for r in roster
        ],
        "code_sha256": {rel: sha256(REPO / rel) for rel in PINNED},
    }
    path = RUNS / f"qwen3_8b_eight_seed_submission_{stamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"\nledger -> {path}", flush=True)


if __name__ == "__main__":
    main()
