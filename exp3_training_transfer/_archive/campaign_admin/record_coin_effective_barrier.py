#!/usr/bin/env python3
"""Record and enforce Coin City canary gating on the effective current roster.

The first Coin City ledger was registered against the original current-campaign
job IDs.  If scheduler-only replacement jobs are later registered, ``afterany``
on the superseded IDs no longer orders Coin City after the effective work.  This
utility updates only the six still-pending canary dependencies, writes a derived
immutable ledger, and replaces only the still-pending Coin City paper finalizer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
COIN = ROOT / "coin_city_structural"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_ids(rows: list[dict], *, label: str, expected: int) -> list[str]:
    result = [str(row["job_id"]) for row in rows]
    if len(result) != expected or len(set(result)) != expected:
        raise AssertionError(f"{label}: expected {expected} distinct job IDs")
    if any(not value.isdigit() for value in result):
        raise AssertionError(f"{label}: non-numeric job ID")
    return result


def pending(job_ids: list[str]) -> None:
    output = subprocess.check_output(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T"], text=True
    )
    states = {}
    for line in output.splitlines():
        if "|" in line:
            job_id, state = line.split("|", 1)
            states[job_id] = state.strip()
    invalid = {job_id: states.get(job_id, "NOT_IN_QUEUE") for job_id in job_ids
               if states.get(job_id) != "PENDING"}
    if invalid:
        raise SystemExit(f"refusing to edit non-pending canaries: {invalid}")


def replace_dependency(command: list[str], dependency: str) -> list[str]:
    result = [token for token in command if not token.startswith("--dependency=")]
    result.insert(-1, f"--dependency={dependency}")
    return result


def submit_finalizer(ledger: Path, dependency_ids: list[str], *, dry_run: bool) -> str:
    command = [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=4", "--mem=16G",
        "--time=01:00:00", "--dependency=afterany:" + ":".join(dependency_ids),
        f"--export=ALL,CAMPAIGN_LEDGER={ledger}",
        str(COIN / "finalize_campaign.sbatch"),
    ]
    print(shlex.join(command), flush=True)
    if dry_run:
        return "DRY_FINALIZER"
    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise AssertionError(f"finalizer submission did not return a numeric ID: {output}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--mechanism-ledger", type=Path, required=True)
    parser.add_argument("--polymarket-ledger", type=Path, required=True)
    parser.add_argument("--recovery-handoff", type=Path, required=True)
    parser.add_argument("--old-finalizer", default="30730393")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = {name: path.resolve() for name, path in {
        "coin_city": args.coin_ledger,
        "mechanism": args.mechanism_ledger,
        "polymarket": args.polymarket_ledger,
        "handoff": args.recovery_handoff,
    }.items()}
    payloads = {name: load(path) for name, path in paths.items()}
    coin = payloads["coin_city"]
    mechanism = payloads["mechanism"]
    polymarket = payloads["polymarket"]
    handoff = payloads["handoff"]

    mechanism_ids = numeric_ids(mechanism["submissions"], label="mechanism", expected=48)
    poly_train_ids = numeric_ids(polymarket["submissions"], label="Polymarket training", expected=6)
    poly_eval_ids = numeric_ids(polymarket["locked_evaluations"], label="Polymarket locked eval", expected=2)
    effective_ids = mechanism_ids + poly_train_ids + poly_eval_ids
    if len(set(effective_ids)) != 56:
        raise AssertionError("effective current roster must contain 56 distinct jobs")

    canary_ids = numeric_ids(coin["canaries"], label="Coin City canaries", expected=6)
    pending(canary_ids)
    dependency = "afterany:" + ":".join(effective_ids)
    for job_id in canary_ids:
        command = ["scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}"]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    derived = copy.deepcopy(coin)
    derived["upstream_dependency"] = {
        "semantics": "afterany",
        "job_ids": effective_ids,
        "effective_roster_ledgers": {
            "mechanism": str(paths["mechanism"].relative_to(REPO)),
            "polymarket": str(paths["polymarket"].relative_to(REPO)),
        },
    }
    for row in derived["canaries"]:
        row["command"] = replace_dependency(list(row["command"]), dependency)
    correction = {
        "recorded_at": timestamp,
        "reason": "scheduler replacements made the original afterany roster non-authoritative",
        "scientific_configuration": "identical",
        "scheduler_change_only": {"canary_dependency": dependency},
        "source_ledger": str(paths["coin_city"].relative_to(REPO)),
        "source_ledger_sha256": sha256(paths["coin_city"]),
        "effective_roster_count": len(effective_ids),
        "canary_job_ids": canary_ids,
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
    }
    derived.setdefault("scheduler_recovery", {})["canary_dependency_barrier"] = correction
    destination = paths["coin_city"].with_name(
        f"{paths['coin_city'].stem}_effectivebarrier_{timestamp}.json"
    )
    if args.dry_run:
        print(f"DRY RUN: would write {destination}")
        return
    destination.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    post_gate_ids = numeric_ids(derived["full_training"], label="Coin City training", expected=21)
    post_gate_ids += numeric_ids(derived["base_evaluations"], label="Coin City base", expected=3)
    finalizer = submit_finalizer(destination, post_gate_ids, dry_run=False)
    subprocess.run(["scancel", args.old_finalizer], check=True)

    revised_handoff = copy.deepcopy(handoff)
    revised_handoff["source_handoff"] = str(paths["handoff"].relative_to(REPO))
    revised_handoff["source_handoff_sha256"] = sha256(paths["handoff"])
    revised_handoff["derived_ledgers"]["coin_city"] = str(destination.relative_to(REPO))
    revised_handoff.setdefault("cancelled_superseded_finalizers", []).append(args.old_finalizer)
    revised_handoff["replacement_finalizers"]["coin_city"] = finalizer
    revised_handoff["canary_barrier_correction"] = correction
    handoff_path = ROOT / "runs" / f"walltime_recovery_effectivebarrier_{timestamp}.json"
    handoff_path.write_text(
        json.dumps(revised_handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"derived Coin City ledger -> {destination}")
    print(f"revised handoff -> {handoff_path}")
    print(f"replacement Coin City finalizer -> {finalizer}")


if __name__ == "__main__":
    main()
