#!/usr/bin/env python3
"""Register the final afterok campaign-to-paper verification job."""
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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-ledger", type=Path, required=True)
    parser.add_argument("--polymarket-ledger", type=Path, required=True)
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument(
        "--supersedes-job-id",
        help="pending c3_verify_paper job replaced by this hash-pinned registration",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = {name: path.resolve() for name, path in {
        "mechanism": args.mechanism_ledger,
        "polymarket": args.polymarket_ledger,
        "coin_city": args.coin_ledger,
        "handoff": args.handoff,
    }.items()}
    handoff = load(paths["handoff"])
    finalizers = handoff.get("replacement_finalizers") or {}
    dependency_ids = [str(finalizers.get(key, "")) for key in ("current", "coin_city")]
    if len(set(dependency_ids)) != 2 or any(not value.isdigit() for value in dependency_ids):
        raise AssertionError("handoff does not identify two distinct finalizers")
    script = ROOT / "verify_paper_handoff.sbatch"
    verifier = ROOT / "verify_paper_handoff.py"
    verifier_hash = sha256(verifier)
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    exports = {
        "MECHANISM_LEDGER": paths["mechanism"],
        "POLY_EXTENSION_LEDGER": paths["polymarket"],
        "COIN_LEDGER": paths["coin_city"],
        "HANDOFF": paths["handoff"],
        "EXPECTED_VERIFIER_SHA256": verifier_hash,
        "EXPECTED_MECHANISM_LEDGER_SHA256": input_hashes["mechanism"],
        "EXPECTED_POLY_EXTENSION_LEDGER_SHA256": input_hashes["polymarket"],
        "EXPECTED_COIN_LEDGER_SHA256": input_hashes["coin_city"],
        "EXPECTED_HANDOFF_SHA256": input_hashes["handoff"],
    }
    superseded = None
    if args.supersedes_job_id:
        if not args.supersedes_job_id.isdigit():
            raise AssertionError("--supersedes-job-id must be numeric")
        status = subprocess.check_output(
            ["scontrol", "show", "job", args.supersedes_job_id, "-o"], text=True
        ).strip()
        if "JobName=c3_verify_paper" not in status or "JobState=PENDING" not in status:
            raise AssertionError(
                "superseded job must be the still-pending c3_verify_paper job"
            )
        superseded = {
            "job_id": args.supersedes_job_id,
            "validated_state": "PENDING",
            "validated_name": "c3_verify_paper",
            "reason": "replacement adds fail-closed verifier and input hash enforcement",
        }
    command = [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=2", "--mem=8G",
        "--time=00:30:00", "--dependency=afterok:" + ":".join(dependency_ids),
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(script),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise AssertionError(f"submission did not return a numeric ID: {output}")
    if superseded is not None:
        subprocess.check_call(["scancel", superseded["job_id"]])
        superseded["cancellation_command"] = ["scancel", superseded["job_id"]]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "registered_at": timestamp,
        "job_id": job_id,
        "dependency_semantics": "afterok",
        "dependency_job_ids": dependency_ids,
        "command": command,
        "script": str(script.relative_to(REPO)),
        "script_sha256": sha256(script),
        "verifier": str(verifier.relative_to(REPO)),
        "verifier_sha256": verifier_hash,
        "inputs": {
            name: {"path": str(path.relative_to(REPO)), "sha256": input_hashes[name]}
            for name, path in paths.items()
        },
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
    }
    if superseded is not None:
        record["superseded_verifier"] = superseded
    revised = copy.deepcopy(handoff)
    revised["parent_handoff"] = str(paths["handoff"].relative_to(REPO))
    revised["parent_handoff_sha256"] = sha256(paths["handoff"])
    revised["paper_verifier"] = record
    destination = ROOT / "runs" / f"paper_verifier_{timestamp}.json"
    destination.write_text(json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"handoff -> {destination}")
    print(f"paper verifier -> {job_id}")


if __name__ == "__main__":
    main()
