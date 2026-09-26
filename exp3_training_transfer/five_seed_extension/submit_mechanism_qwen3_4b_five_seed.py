#!/usr/bin/env python3
"""Submit the Qwen3-4B mechanism-family added seeds 45-46.

Protocol `exp3_mechanism_qwen3_4b_five_seed_v1`
(`MECHANISM_QWEN3_4B_FIVE_SEED_AMENDMENT.md`): 2 disclosures x 2 arms x 2 added
seeds = 8 training jobs, taking those four cells to G=5.

Every generated environment is checked against the registered seed-42
submission recovered from the mechanism ledger and must differ only in SEED.
Endpoint existence is resolved through the registered aggregate, never by
globbing `reports/` -- there are up to seven attempt directories per cell.

Dry run by default; pass --submit to submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MECH = REPO / "exp3_training_transfer/mechanism_family"
RUNS = ROOT / "runs"

PROTOCOL = "exp3_mechanism_qwen3_4b_five_seed_v1"
# Default; --model selects. Qwen3-8B is added under
# exp3_contradiction_resolution_v1 because its disclosed contrast is -0.125
# with 0/3 seeds positive, opposite to Qwen3-4B in the same roster.
MODEL_KEY = "qwen3_4b"
SNAPSHOTS = {
    "qwen3_4b": ".runtime/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
                "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    "qwen3_8b": ".runtime/hf_home/hub/models--Qwen--Qwen3-8B/"
                "snapshots/b968826d9c46dd6066d109eabc6255188de91218",
}
DISCLOSURES = ("disclosed", "undisclosed")
ARMS = ("causal_family", "population_prior")
ADDED_SEEDS = (45, 46)

AMENDMENT = ("exp3_training_transfer/five_seed_extension/"
             "MECHANISM_QWEN3_4B_FIVE_SEED_AMENDMENT.md")
WRAPPER = "exp3_training_transfer/mechanism_family/mechanism_rl.sh"
AGGREGATE = MECH / "reports/c3_mechanism_full_aggregate.json"

# Recorded rather than inherited from the wrapper default, so the value that
# ran is in the ledger. Matches mechanism_rl.sh's own default of 999999.
EXPLICIT = {"SAVE_STEPS": "999999"}

ALLOCATION = ["--parsable", "--partition=all", "--gres=gpu:a6000:2",
              "--cpus-per-task=8", "--mem=100G", "--time=20:00:00",
              "--exclude=node206"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mechanism_ledger() -> dict:
    for path in sorted(MECH.glob("runs/c3_mechanism_*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The effective ledger records the registered roster under "submissions"
        # and carries the C3 protocol id; earlier drafts in runs/ do not.
        if payload.get("submissions") and "mechanism" in str(payload.get("protocol", "")):
            return payload | {"__path__": str(path)}
    raise SystemExit("no mechanism-family ledger found")


def seed42_environments(ledger: dict) -> dict[tuple[str, str], dict[str, str]]:
    """{(disclosure, arm): exported env} for the registered Qwen3-4B seed 42."""
    found: dict[tuple[str, str], dict[str, str]] = {}

    def walk(node):
        if isinstance(node, dict):
            if "command" in node:
                joined = " ".join(node["command"])
                if "--export=ALL," in joined:
                    chunk = joined.split("--export=ALL,", 1)[1].split(" ", 1)[0]
                    env = {k: v for k, _, v in
                           (t.partition("=") for t in chunk.split(","))}
                    if (env.get("MODEL_KEY") == MODEL_KEY and env.get("SEED") == "42"
                            and env.get("ARM") in ARMS
                            and env.get("DISCLOSURE") in DISCLOSURES):
                        found.setdefault((env["DISCLOSURE"], env["ARM"]), env)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(ledger)
    missing = [(d, a) for d in DISCLOSURES for a in ARMS if (d, a) not in found]
    if missing:
        raise SystemExit(f"ledger lacks registered seed-42 submissions for {missing}")
    return found


def registered_cells() -> set[tuple[str, str, int]]:
    payload = json.loads(AGGREGATE.read_text())
    out = set()
    for rel in payload["score_files"]:
        m = re.search(rf"reports/(disclosed|undisclosed)_(causal_family|population_prior)_"
                      rf"{MODEL_KEY}_s(\d+)_", rel)
        if m:
            out.add((m.group(1), m.group(2), int(m.group(3))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--model", default=MODEL_KEY, choices=sorted(SNAPSHOTS))
    parser.add_argument("--amendment", default=AMENDMENT,
                        help="amendment file this submission is recorded under")
    parser.add_argument("--disclosure", choices=DISCLOSURES, action="append",
                        dest="disclosures",
                        help="restrict to these disclosures; repeatable. Qwen3-8B "
                             "undisclosed has a minimum G of 56 and is not worth "
                             "extending, so that roster is disclosed-only.")
    args = parser.parse_args()
    globals()["MODEL_KEY"] = args.model
    globals()["AMENDMENT"] = args.amendment

    for rel in (AMENDMENT, WRAPPER):
        if not (REPO / rel).is_file():
            raise SystemExit(f"missing {rel}")
    snapshot = REPO / SNAPSHOTS[MODEL_KEY]
    if not snapshot.is_dir():
        raise SystemExit(f"missing pinned offline snapshot: {snapshot}")

    ledger = mechanism_ledger()
    parents = seed42_environments(ledger)
    already = registered_cells()

    selected = tuple(args.disclosures) if args.disclosures else DISCLOSURES
    roster = []
    for disclosure in selected:
        for arm in ARMS:
            parent = parents[(disclosure, arm)]
            for seed in ADDED_SEEDS:
                if (disclosure, arm, seed) in already:
                    print(f"skip  {disclosure}/{arm} s{seed}: already registered")
                    continue
                env = dict(parent) | {"SEED": str(seed)} | EXPLICIT
                drift = {k for k in parent if k in env and parent[k] != env[k]}
                if drift - {"SEED"}:
                    raise SystemExit(
                        f"{disclosure}/{arm}: unexpected drift from seed 42: "
                        f"{ {k: (parent[k], env[k]) for k in drift - {'SEED'}} }")
                if set(env) - set(parent) != set(EXPLICIT):
                    raise SystemExit(
                        f"{disclosure}/{arm}: unexpected extra keys "
                        f"{sorted(set(env) - set(parent) - set(EXPLICIT))}")
                export = "ALL," + ",".join(f"{k}={v}" for k, v in sorted(env.items()))
                roster.append({
                    "disclosure": disclosure, "arm": arm, "seed": seed,
                    "environment": env,
                    "command": ["sbatch", *ALLOCATION, f"--export={export}",
                                str(MECH / "mechanism_rl.sh")],
                })
    print(f"environment verified identical to {MODEL_KEY} seed 42 except SEED "
          f"(plus explicit {sorted(EXPLICIT)})", flush=True)

    if not roster:
        print("nothing to submit")
        return
    for record in roster:
        print(shlex.join(record["command"]), flush=True)
    print(f"\n{len(roster)} job(s); submit={args.submit}", flush=True)
    if not args.submit:
        return

    for record in roster:
        out = subprocess.check_output(record["command"], cwd=REPO, text=True).strip()
        job_id = out.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch output: {out!r}")
        record["job_id"] = job_id
        print(f"submitted {job_id}  {record['disclosure']}/{record['arm']} "
              f"s{record['seed']}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUNS.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": stamp,
        "kind": "mechanism_qwen3_4b_five_seed_submission",
        "protocol": PROTOCOL,
        "parent_protocol": "exp3_exp4_five_seed_extension_v1",
        "model_key": MODEL_KEY,
        "disclosures": list(selected),
        "arms": list(ARMS),
        "added_seeds": list(ADDED_SEEDS),
        "combined_seed_roster": [42, 43, 44, *ADDED_SEEDS],
        "explicit_environment": EXPLICIT,
        "allocation": ALLOCATION,
        "submitted": True,
        "scientific_hyperparameters_changed": False,
        "training_jobs": roster,
        "train_script_sha256": sha256(REPO / WRAPPER),
        "code_sha256": {rel: sha256(REPO / rel) for rel in (AMENDMENT, WRAPPER)}
        | {"exp3_training_transfer/five_seed_extension/"
           "submit_mechanism_qwen3_4b_five_seed.py": sha256(Path(__file__))},
    }
    path = RUNS / f"mechanism_qwen3_4b_five_seed_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nledger -> {path}", flush=True)


if __name__ == "__main__":
    main()
