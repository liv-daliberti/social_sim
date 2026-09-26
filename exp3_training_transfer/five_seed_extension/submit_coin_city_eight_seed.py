#!/usr/bin/env python3
"""Submit Coin City added seeds 45-49 for a model, under its eight-seed amendment.

Generalized successor to `submit_qwen3_8b_eight_seed.py`. That script is kept
on disk unchanged because its hash is pinned in the Qwen3-8B submission
ledgers; this one supersedes it for all new work and handles Qwen3-8B too.

Per-model amendments:
  qwen3_8b  exp3_qwen3_8b_coin_city_eight_seed_v1  QWEN3_8B_EIGHT_SEED_AMENDMENT.md
  qwen3_4b  exp3_qwen3_4b_coin_city_eight_seed_v1  QWEN3_4B_EIGHT_SEED_AMENDMENT.md

The per-job environment and sbatch allocation come from `launch.py`'s own
`coin_env` and `gpu_command`, so what is submitted is the frozen parent
protocol code with a different seed tuple, not a retyped copy. Every generated
environment is checked against the parent ledger's actual seed-42 submission
and must differ only in SEED and the derived TAG.

Dry run by default; pass --submit to submit.

    python submit_coin_city_eight_seed.py --model qwen3_4b
    python submit_coin_city_eight_seed.py --model qwen3_4b --submit
    python submit_coin_city_eight_seed.py --model qwen3_8b --only 48 --arm causal --submit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import COIN, REPO, RUNS, coin_env, gpu_command  # noqa: E402

ADDED_SEEDS = (45, 46, 47, 48, 49)
ARMS = ("causal", "population_prior")
GREEDY_ROWS = 1440
STOCHASTIC_ROWS = 1440 * 5

# train.sh was modified on 2026-08-28, after seeds 42--44 ran: --save_steps
# went from 999999 to a 40 default and --save_ckpt was added, both of which
# apply silently because coin_env sets neither. Passing these two values
# reproduces the emitted oat command line of the wrapper at HEAD (committed
# 2026-08-22), which is what seeds 42--44 actually trained under. They are
# output/checkpointing flags, not scientific parameters.
WRAPPER_RESTORE = {"SAVE_STEPS": "999999", "SAVE_CKPT": "0"}

PARENT_PROTOCOL = "exp3_exp4_five_seed_extension_v1"
PARENT_SHA = "4c485b67a497fb36a3b866a964090af067e761bcf9512aca7e9723ebcf9d89ed"
TRAIN_SH = "exp3_training_transfer/coin_city_structural/train.sh"
TRAIN_SH_SHA = "7ca287c389ff4f18678de9e381e5de6b0064cdd2635eebbfab3a749476685a21"

MODELS = {
    "qwen3_8b": {
        "amendment": "exp3_qwen3_8b_coin_city_eight_seed_v1",
        "amendment_file": "exp3_training_transfer/five_seed_extension/"
                          "QWEN3_8B_EIGHT_SEED_AMENDMENT.md",
        "amendment_sha": "5d12d8a64922e73f5d63e55cdff8421d19cd54a59075567e2f8d5d5221810179",
        "snapshot": ".runtime/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/"
                    "b968826d9c46dd6066d109eabc6255188de91218",
    },
    "llama3_1_8b": {
        # Added under exp3_contradiction_resolution_v1 to G=5, not G=8: this
        # cell's in-distribution contrast is -0.351 with 0/3 seeds positive and
        # a minimum G of 4, so five seeds settle it.
        "amendment": "exp3_contradiction_resolution_v1",
        "amendment_file": "exp3_training_transfer/five_seed_extension/"
                          "CONTRADICTION_RESOLUTION_AMENDMENT.md",
        "amendment_sha": "5294077c7e5e556423a47ba3b89af15efa0844b5751626120b65eaad752f6150",
        "snapshot": ".runtime/hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
                    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "qwen3_4b": {
        "amendment": "exp3_qwen3_4b_coin_city_eight_seed_v1",
        "amendment_file": "exp3_training_transfer/five_seed_extension/"
                          "QWEN3_4B_EIGHT_SEED_AMENDMENT.md",
        "amendment_sha": "afdb6f1dc91a7fbf687a6d0eb24d5250733fa210ed2dfa04fff4dce2198387d7",
        "snapshot": ".runtime/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
                    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins(model_key: str) -> dict[str, str]:
    spec = MODELS[model_key]
    pinned = {
        "exp3_training_transfer/five_seed_extension/PROTOCOL.md": PARENT_SHA,
        spec["amendment_file"]: spec["amendment_sha"],
        TRAIN_SH: TRAIN_SH_SHA,
    }
    for rel, expected in pinned.items():
        actual = sha256(REPO / rel)
        if actual != expected:
            raise SystemExit(f"hash drift, refusing to submit:\n  {rel}\n"
                             f"  expected {expected}\n  actual   {actual}")
    return pinned


def verify_inputs(model_key: str) -> None:
    for arm in ARMS:
        for split in ("train", "heldout"):
            path = COIN / "data" / arm / split
            if not path.is_dir():
                raise SystemExit(f"missing dataset: {path}")
    snapshot = REPO / MODELS[model_key]["snapshot"]
    if not snapshot.is_dir():
        raise SystemExit(f"missing pinned offline snapshot: {snapshot}")


def verify_environment_matches_parent(model_key: str, restore: bool) -> None:
    """Generated env must equal the parent ledger's seed-42 env but for SEED/TAG."""
    ledger = None
    for path in sorted((COIN / "runs").glob("coin_city_structural_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("protocol") == "coin_city_structural_transfer_v1":
            ledger = payload
            break
    if ledger is None:
        raise SystemExit("no effective Coin City structural-transfer ledger found")

    wanted = {(row["arm"], int(row["seed"])): str(row["job_id"])
              for row in ledger.get("full_training", [])
              if row.get("model") == model_key and row.get("kind") == "confirmatory"}
    blob = json.dumps(ledger)
    for arm in ARMS:
        job_id = wanted.get((arm, 42))
        if job_id is None:
            raise SystemExit(f"parent ledger lacks {model_key} {arm} seed 42")
        marker = f'--export=ALL,'
        old: dict[str, str] = {}

        def walk(node):
            if isinstance(node, dict):
                if str(node.get("job_id", "")) == job_id and "command" in node:
                    joined = " ".join(node["command"])
                    if marker in joined:
                        chunk = joined.split(marker, 1)[1].split(" ", 1)[0]
                        for token in chunk.split(","):
                            key, _, value = token.partition("=")
                            old[key] = value
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(ledger)
        if not old:
            raise SystemExit(f"could not recover seed-42 environment for {arm} "
                             f"from the parent ledger (job {job_id})")
        new = coin_env(model_key, arm, 45)
        if restore:
            new = new | WRAPPER_RESTORE
            # These keys are absent from the seed-42 submission precisely
            # because the wrapper hardcoded their values then; compare against
            # the parent as if they had been passed explicitly.
            old = old | WRAPPER_RESTORE
        if set(old) != set(new):
            raise SystemExit(
                f"{model_key} {arm}: environment key set differs from seed 42\n"
                f"  only in seed 42: {sorted(set(old) - set(new))}\n"
                f"  only in added  : {sorted(set(new) - set(old))}")
        differing = {k for k in old if old[k] != new[k]}
        if differing - {"SEED", "TAG"}:
            raise SystemExit(
                f"{model_key} {arm}: unexpected environment drift from seed 42: "
                f"{ {k: (old[k], new[k]) for k in differing - {'SEED', 'TAG'}} }")
    print(f"environment verified identical to {model_key} seed 42 except SEED/TAG",
          flush=True)


def cell_state(model_key: str, arm: str, seed: int) -> tuple[str, list[pathlib.Path]]:
    dirs = sorted((COIN / "reports").glob(f"{arm}_{model_key}_s{seed}_*"))
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


def partition_roster(model_key: str, roster: list[dict]) -> tuple[list[dict], list[str]]:
    submit, skipped = [], []
    for record in roster:
        state, dirs = cell_state(model_key, record["arm"], record["seed"])
        label = f"{record['arm']} s{record['seed']}"
        if state == "complete":
            skipped.append(f"{label}: complete, skipping ({dirs[0].name})")
        elif state == "ambiguous":
            raise SystemExit(
                f"{label}: a complete endpoint coexists with other attempt "
                f"directories; move the failed ones under reports/failed_attempts/:\n  "
                + "\n  ".join(d.name for d in dirs))
        elif state == "partial":
            raise SystemExit(
                f"{label}: an attempt directory exists without complete endpoint "
                f"files:\n  " + "\n  ".join(d.name for d in dirs)
                + "\n\nCheck `sacct` first. If that job is still PENDING or RUNNING, "
                  "this cell is in flight and must be left alone -- do NOT move the "
                  "directory, it is the live job's output path. Only if the job has "
                  "reached a terminal state is this a failed attempt, in which case "
                  "move it under reports/failed_attempts/ and retry.")
        else:
            submit.append(record)
    return submit, skipped


def records(model_key: str, restore: bool = False) -> list[dict]:
    out = []
    for arm in ARMS:
        for seed in ADDED_SEEDS:
            environment = coin_env(model_key, arm, seed)
            if restore:
                environment = environment | WRAPPER_RESTORE
            command = gpu_command(
                COIN / "train.sh", environment,
                gpus=2, cpus=8, memory="100G", walltime="30:00:00",
                partition="cs,all", account="allcs", qos="medium",
            )
            out.append({"arm": arm, "seed": seed, "identity": f"{model_key}/{arm}",
                        "environment": environment, "command": command})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--only", type=int, nargs="+", metavar="SEED",
                        help="restrict to these added seeds (infrastructural retry only)")
    parser.add_argument("--arm", choices=ARMS, action="append", dest="arms")
    parser.add_argument("--added-seeds", type=int, nargs="+",
                        help="override the default 45-49 roster (e.g. 45 46 for a "
                             "G=5 extension under a different amendment)")
    parser.add_argument("--restore-parent-wrapper", action="store_true",
                        help="pass SAVE_STEPS=999999 SAVE_CKPT=0 so the emitted oat "
                             "command matches the wrapper seeds 42-44 trained under")
    args = parser.parse_args()

    model_key = args.model
    pinned = verify_pins(model_key)
    verify_inputs(model_key)
    verify_environment_matches_parent(model_key, args.restore_parent_wrapper)

    if args.added_seeds:
        globals()["ADDED_SEEDS"] = tuple(sorted(set(args.added_seeds)))
    roster = records(model_key, args.restore_parent_wrapper)
    if args.only:
        roster = [r for r in roster if r["seed"] in set(args.only)]
    if args.arms:
        roster = [r for r in roster if r["arm"] in set(args.arms)]
    if not roster:
        raise SystemExit("seed/arm filters matched no roster entry")

    roster, skipped = partition_roster(model_key, roster)
    for line in skipped:
        print(f"skip  {line}", flush=True)
    if not roster:
        print("\nnothing to submit; every selected cell already has a complete endpoint")
        return

    for record in roster:
        print(shlex.join(record["command"]), flush=True)
    print(f"\n{len(roster)} job(s) for {model_key}; submit={args.submit}", flush=True)
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
        "kind": f"{model_key}_eight_seed_submission",
        "protocol": MODELS[model_key]["amendment"],
        "parent_protocol": PARENT_PROTOCOL,
        "model_key": model_key,
        "added_seeds": list(ADDED_SEEDS),
        "combined_seed_roster": [42, 43, 44, *ADDED_SEEDS],
        "train_script_sha256": sha256(REPO / TRAIN_SH),
        "arms": list(ARMS),
        "submitted": True,
        "scientific_hyperparameters_changed": False,
        "parent_wrapper_restored": args.restore_parent_wrapper,
        "wrapper_restore_env": WRAPPER_RESTORE if args.restore_parent_wrapper else None,
        "partition_requested": "cs,all",
        "effective_partition": "cs",
        "training_jobs": roster,
        "code_sha256": {rel: sha256(REPO / rel) for rel in pinned}
        | {"exp3_training_transfer/five_seed_extension/submit_coin_city_eight_seed.py":
           sha256(Path(__file__))},
    }
    path = RUNS / f"{model_key}_eight_seed_submission_{stamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"\nledger -> {path}", flush=True)


if __name__ == "__main__":
    main()
