#!/usr/bin/env python3
"""Replace one silently stalled current-campaign cell and rewire its handoff.

The source allocation must still be RUNNING, have a stale log, and contain the
registered NCCL/bind failure signature.  The replacement is submitted before
anything is cancelled and is scientifically byte-identical to the source row;
only an ``afterany`` dependency on the stalled allocation is added.  The script
then derives immutable mechanism/Coin ledgers, updates the six pending Coin
canary barriers, replaces both paper finalizers, and delegates creation of the
hash-pinned terminal verifier to ``register_paper_verifier.py``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MECHANISM = ROOT / "mechanism_family"
COIN = ROOT / "coin_city_structural"
FAILURE_MARKERS = (
    "torch.distributed.DistBackendError",
    "Call to bind failed : Cannot assign requested address",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def job_status(job_id: str) -> str:
    return subprocess.check_output(
        ["scontrol", "show", "job", job_id, "-o"], text=True
    ).strip()


def require_job(job_id: str, *, name: str, state: str) -> str:
    status = job_status(job_id)
    if f"JobName={name}" not in status or f"JobState={state}" not in status:
        raise AssertionError(f"job {job_id} must be {name} in {state}: {status}")
    return status


def submit(command: list[str]) -> str:
    print(shlex.join(command), flush=True)
    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise AssertionError(f"submission did not return a numeric job ID: {output}")
    return job_id


def add_afterany(command: list[str], source_job_id: str) -> list[str]:
    result: list[str] = []
    dependencies: list[str] = []
    for token in command:
        if token.startswith("--dependency="):
            dependencies.extend(token.split("=", 1)[1].split(","))
        else:
            result.append(token)
    dependency = f"afterany:{source_job_id}"
    if dependency not in dependencies:
        dependencies.append(dependency)
    result.insert(-1, "--dependency=" + ",".join(dependencies))
    return result


def replace_dependency_token(command: list[str], old: str, new: str) -> list[str]:
    result = []
    found = False
    for token in command:
        if token.startswith("--dependency="):
            ids = token.split("=", 1)[1].split(":")
            found |= old in ids
            token = token.replace(old, new)
        result.append(token)
    if not found:
        raise AssertionError(f"canary command did not contain source job {old}")
    return result


def roster_ids(mechanism: dict, poly: dict) -> list[str]:
    ids = [
        str(row["job_id"])
        for row in mechanism.get("submissions", [])
        if row.get("kind") in {"training", "base_evaluation"}
    ]
    ids += [str(row["job_id"]) for row in poly.get("submissions", [])]
    ids += [str(row["job_id"]) for row in poly.get("locked_evaluations", [])]
    if len(ids) != 56 or len(set(ids)) != 56 or any(not value.isdigit() for value in ids):
        raise AssertionError("effective current roster must contain 56 distinct numeric IDs")
    return ids


def coin_endpoint_ids(coin: dict) -> list[str]:
    ids = [str(row["job_id"]) for row in coin.get("full_training", [])]
    ids += [str(row["job_id"]) for row in coin.get("base_evaluations", [])]
    if len(ids) != 24 or len(set(ids)) != 24 or any(not value.isdigit() for value in ids):
        raise AssertionError("Coin finalizer roster must contain 24 distinct endpoints")
    return ids


def finalizer_command(script: Path, dependencies: list[str], exports: dict[str, str]) -> list[str]:
    return [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=4", "--mem=16G",
        "--time=01:00:00", "--dependency=afterany:" + ":".join(dependencies),
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(script),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-ledger", type=Path, required=True)
    parser.add_argument("--polymarket-ledger", type=Path, required=True)
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--stalled-job-id", required=True)
    parser.add_argument("--old-current-finalizer", required=True)
    parser.add_argument("--old-coin-finalizer", required=True)
    parser.add_argument("--old-verifier", required=True)
    parser.add_argument("--minimum-stale-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = {
        "mechanism": args.mechanism_ledger.resolve(),
        "polymarket": args.polymarket_ledger.resolve(),
        "coin_city": args.coin_ledger.resolve(),
        "handoff": args.handoff.resolve(),
    }
    mechanism, poly, coin, handoff = (load(paths[key]) for key in
                                      ("mechanism", "polymarket", "coin_city", "handoff"))
    source_rows = [row for row in mechanism.get("submissions", [])
                   if str(row.get("job_id")) == args.stalled_job_id]
    if len(source_rows) != 1 or source_rows[0].get("kind") != "training":
        raise AssertionError("stalled job must identify exactly one mechanism training row")
    source = source_rows[0]
    source_status = require_job(args.stalled_job_id, name="c3_mech", state="RUNNING")
    require_job(args.old_current_finalizer, name="c3_current_paper", state="PENDING")
    require_job(args.old_coin_finalizer, name="c3_coin_paper", state="PENDING")
    require_job(args.old_verifier, name="c3_verify_paper", state="PENDING")

    canary_ids = [str(row["job_id"]) for row in coin.get("canaries", [])]
    if len(canary_ids) != 6 or len(set(canary_ids)) != 6:
        raise AssertionError("Coin ledger must contain exactly six canaries")
    for job_id in canary_ids:
        require_job(job_id, name="c3_coin", state="PENDING")

    source_log = MECHANISM / "logs" / f"train_{args.stalled_job_id}.out"
    if not source_log.is_file():
        raise AssertionError(f"missing stalled-job log: {source_log}")
    age = time.time() - source_log.stat().st_mtime
    text = source_log.read_text(encoding="utf-8", errors="replace")
    if age < args.minimum_stale_seconds:
        raise AssertionError(f"stalled-job log is only {age:.0f}s old")
    if any(marker not in text for marker in FAILURE_MARKERS):
        raise AssertionError("stalled-job log lacks the required NCCL/bind failure markers")

    replacement_command = add_afterany(list(source["command"]), args.stalled_job_id)
    if args.dry_run:
        print(json.dumps({
            "validated_source": args.stalled_job_id,
            "cell": {key: source.get(key) for key in ("disclosure", "model", "arm", "seed")},
            "stale_seconds": int(age),
            "replacement_command": replacement_command,
            "canaries_to_rewire": canary_ids,
            "finalizers_to_replace": [args.old_current_finalizer, args.old_coin_finalizer],
            "verifier_to_replace": args.old_verifier,
        }, indent=2, sort_keys=True))
        return

    replacement_id = submit(replacement_command)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery = {
        "registered_at": timestamp,
        "reason": "source allocation remained RUNNING after an NCCL communicator bind timeout and produced no further output",
        "scientific_configuration": "identical",
        "scheduler_change_only": {"dependency_added": f"afterany:{args.stalled_job_id}"},
        "source_job_id": args.stalled_job_id,
        "replacement_job_id": replacement_id,
        "source_scheduler_snapshot": source_status,
        "source_log": str(source_log.relative_to(REPO)),
        "source_log_sha256_pre_cancel": sha256(source_log),
        "source_log_stale_seconds": int(age),
        "failure_markers": list(FAILURE_MARKERS),
        "cell": {key: source.get(key) for key in ("disclosure", "model", "arm", "seed")},
        "replacement_command": replacement_command,
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
    }

    derived_mechanism = copy.deepcopy(mechanism)
    derived_row = next(row for row in derived_mechanism["submissions"]
                       if str(row.get("job_id")) == args.stalled_job_id)
    derived_row["job_id"] = replacement_id
    derived_row["replaces_job_id"] = args.stalled_job_id
    derived_row["command"] = replacement_command
    derived_mechanism.setdefault("runtime_recoveries", []).append(recovery)
    mechanism_path = paths["mechanism"].with_name(
        f"{paths['mechanism'].stem}_runtimefix_{timestamp}.json"
    )
    write_json(mechanism_path, derived_mechanism)

    current_ids = roster_ids(derived_mechanism, poly)
    if args.stalled_job_id in current_ids or replacement_id not in current_ids:
        raise AssertionError("derived current roster did not replace the stalled job exactly")
    dependency = "afterany:" + ":".join(current_ids)

    derived_coin = copy.deepcopy(coin)
    upstream = derived_coin.get("upstream_dependency") or {}
    upstream_ids = [replacement_id if str(value) == args.stalled_job_id else str(value)
                    for value in upstream.get("job_ids", [])]
    if len(upstream_ids) != 56 or len(set(upstream_ids)) != 56:
        raise AssertionError("derived Coin upstream roster is not the exact 56-job barrier")
    upstream["job_ids"] = upstream_ids
    upstream["effective_roster_ledgers"] = {
        "mechanism": str(mechanism_path.relative_to(REPO)),
        "polymarket": str(paths["polymarket"].relative_to(REPO)),
    }
    for row in derived_coin["canaries"]:
        row["command"] = replace_dependency_token(
            list(row["command"]), args.stalled_job_id, replacement_id
        )
    barrier = ((derived_coin.get("scheduler_recovery") or {})
               .get("canary_dependency_barrier") or {})
    barrier.setdefault("prior_runtime_dependency", barrier.get("scheduler_change_only", {}))
    barrier["scheduler_change_only"] = {"canary_dependency": dependency}
    barrier["runtime_replacement_job_id"] = replacement_id
    barrier["runtime_source_job_id"] = args.stalled_job_id
    derived_coin.setdefault("runtime_recoveries", []).append(recovery)
    coin_path = paths["coin_city"].with_name(
        f"{paths['coin_city'].stem}_runtimefix_{timestamp}.json"
    )
    write_json(coin_path, derived_coin)

    for job_id in canary_ids:
        subprocess.check_call([
            "scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}"
        ])
        status = job_status(job_id)
        if replacement_id not in status or args.stalled_job_id in status:
            raise AssertionError(f"canary dependency update failed for {job_id}: {status}")

    current_finalizer = submit(finalizer_command(
        ROOT / "finalize_current_campaign.sbatch", current_ids,
        {"MECHANISM_LEDGER": str(mechanism_path),
         "POLY_EXTENSION_LEDGER": str(paths["polymarket"])},
    ))
    analysis = handoff.get("coin_analysis_extension") or handoff.get("coin_analysis_repair") or {}
    renderer_hash = str(analysis.get("renderer_sha256") or "")
    if len(renderer_hash) != 64:
        raise AssertionError("active handoff lacks the pinned Coin renderer hash")
    coin_finalizer = submit(finalizer_command(
        COIN / "finalize_campaign.sbatch", coin_endpoint_ids(derived_coin),
        {"CAMPAIGN_LEDGER": str(coin_path), "EXPECTED_RENDERER_SHA256": renderer_hash},
    ))

    subprocess.check_call([
        "scancel", args.stalled_job_id, args.old_current_finalizer, args.old_coin_finalizer
    ])
    recovery["cancellation_command"] = [
        "scancel", args.stalled_job_id, args.old_current_finalizer, args.old_coin_finalizer
    ]
    recovery["replacement_finalizers"] = {
        "current": current_finalizer, "coin_city": coin_finalizer
    }
    recovery["canary_dependency"] = dependency
    recovery["derived_ledgers"] = {
        "mechanism": str(mechanism_path.relative_to(REPO)),
        "coin_city": str(coin_path.relative_to(REPO)),
        "polymarket": str(paths["polymarket"].relative_to(REPO)),
    }

    revised = copy.deepcopy(handoff)
    revised["parent_handoff"] = str(paths["handoff"].relative_to(REPO))
    revised["parent_handoff_sha256"] = sha256(paths["handoff"])
    revised["derived_ledgers"] = recovery["derived_ledgers"]
    revised["replacement_finalizers"] = recovery["replacement_finalizers"]
    revised.setdefault("runtime_recoveries", []).append(recovery)
    if revised.get("coin_analysis_extension"):
        revised["coin_analysis_extension"]["coin_ledger"] = str(coin_path.relative_to(REPO))
        revised["coin_analysis_extension"]["coin_ledger_sha256"] = sha256(coin_path)
        revised["coin_analysis_extension"]["replacement_finalizer"] = coin_finalizer
    repair_handoff = ROOT / "runs" / f"runtime_recovery_{timestamp}.json"
    write_json(repair_handoff, revised)

    verifier_output = subprocess.check_output([
        sys.executable, str(ROOT / "register_paper_verifier.py"),
        "--mechanism-ledger", str(mechanism_path),
        "--polymarket-ledger", str(paths["polymarket"]),
        "--coin-ledger", str(coin_path),
        "--handoff", str(repair_handoff),
        "--supersedes-job-id", args.old_verifier,
    ], cwd=REPO, text=True)
    print(verifier_output, end="")
    print(f"replacement training job -> {replacement_id}")
    print(f"derived mechanism ledger -> {mechanism_path}")
    print(f"derived Coin ledger -> {coin_path}")
    print(f"runtime-recovery handoff -> {repair_handoff}")
    print(f"current finalizer -> {current_finalizer}")
    print(f"Coin finalizer -> {coin_finalizer}")


if __name__ == "__main__":
    main()
