#!/usr/bin/env python3
"""Replace at-risk 8B jobs with scientifically identical 30-hour submissions.

All replacements are submitted with ``afterany`` on their source job before any
source is cancelled.  Derived ledgers preserve the source hash and exact mapping.
The operation therefore fails safe: a submission error leaves every original
untouched, and renderers consume only the explicit derived ledgers.
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
MECHANISM = ROOT / "mechanism_family"
POLY = ROOT / "polymarket"
COIN = ROOT / "coin_city_structural"
AT_RISK_MODELS = {"qwen3_8b", "llama3_1_8b"}
NEW_TIME_LIMIT = "1-06:00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def submit(command: list[str], *, dry_run: bool) -> str:
    print(shlex.join(command), flush=True)
    if dry_run:
        return f"DRY{len(command)}"
    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise AssertionError(f"submission did not return a numeric job ID: {output}")
    return job_id


def scheduler_states(job_ids: list[str]) -> dict[str, str]:
    output = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
        "-o", "JobIDRaw,State",
    ], text=True)
    states = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        if "." not in job_id:
            states[job_id] = state.split()[0].split("+", 1)[0]
    queue = subprocess.check_output([
        "squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T"
    ], text=True)
    for line in queue.splitlines():
        if "|" in line:
            job_id, state = line.split("|", 1)
            states[job_id] = state.strip()
    return states


def repaired_command(command: list[str], old_job_id: str, *, walltime: str,
                     prerequisite: str | None = None) -> list[str]:
    result = []
    dependency_groups = []
    for token in command:
        if token.startswith("--time="):
            continue
        if token.startswith("--dependency="):
            dependency_groups.append(token.split("=", 1)[1])
            continue
        result.append(token)
    result.insert(-1, f"--time={walltime}")
    dependency_groups.append(f"afterany:{old_job_id}")
    if prerequisite:
        dependency_groups.insert(0, prerequisite)
    result.insert(-1, "--dependency=" + ",".join(dependency_groups))
    return result


def replace_rows(ledger: dict, rows: list[dict], mapping: dict[str, dict]) -> None:
    for row in rows:
        old = str(row["job_id"])
        replacement = mapping[old]
        row["job_id"] = replacement["replacement_job_id"]
        row["replaces_job_id"] = old
        row["command"] = replacement["command"]
        row["scheduler_time_limit_override"] = NEW_TIME_LIMIT


def training_candidates(mechanism: dict, poly: dict, coin: dict) -> tuple[list, list, list]:
    mechanism_rows = [
        row for row in mechanism["submissions"]
        if row.get("kind") == "training" and row.get("model") in AT_RISK_MODELS
    ]
    poly_rows = [row for row in poly["submissions"] if row.get("model") in AT_RISK_MODELS]
    coin_rows = [
        row for row in coin["full_training"] if row.get("model") in AT_RISK_MODELS
    ]
    if (len(mechanism_rows), len(poly_rows), len(coin_rows)) != (24, 6, 12):
        raise AssertionError(
            "expected 24 mechanism, 6 Polymarket, and 12 Coin City at-risk cells"
        )
    return mechanism_rows, poly_rows, coin_rows


def update_export(command: list[str], replacements: dict[str, str]) -> list[str]:
    output = []
    for token in command:
        for old, new in replacements.items():
            token = token.replace(old, new)
        output.append(token)
    return output


def locked_eval_command(item: dict, model_training: list[dict], mapping: dict[str, dict]) -> tuple[list[str], str]:
    old_eval = str(item["job_id"])
    seed_rows = sorted(model_training, key=lambda row: int(row["seed"]))
    new_ids = [mapping[str(row["job_id"])]["replacement_job_id"] for row in seed_rows]
    id_map = {str(row["job_id"]): new for row, new in zip(seed_rows, new_ids)}
    command = update_export(list(item["command"]), id_map)
    command = [token for token in command if not token.startswith("--dependency=")]
    command.insert(
        -1,
        "--dependency=afterok:" + ":".join(new_ids) + f",afterany:{old_eval}",
    )
    adapter_spec = ";".join(
        f"{row['seed']}:{new}" for row, new in zip(seed_rows, new_ids)
    )
    return command, adapter_spec


def write_derived(source: Path, payload: dict, timestamp: str, label: str) -> Path:
    destination = source.with_name(f"{source.stem}_{label}_{timestamp}.json")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def finalizer_command(script: Path, dependency_ids: list[str], exports: dict[str, str]) -> list[str]:
    return [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=4", "--mem=16G",
        "--time=01:00:00", "--dependency=afterany:" + ":".join(dependency_ids),
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(script),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-ledger", type=Path, required=True)
    parser.add_argument("--polymarket-ledger", type=Path, required=True)
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--old-current-finalizer", default="30728529")
    parser.add_argument("--old-coin-finalizer", default="30728530")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_paths = {
        "mechanism": args.mechanism_ledger.resolve(),
        "polymarket": args.polymarket_ledger.resolve(),
        "coin_city": args.coin_ledger.resolve(),
    }
    ledgers = {name: load(path) for name, path in source_paths.items()}
    mechanism_rows, poly_rows, coin_rows = training_candidates(
        ledgers["mechanism"], ledgers["polymarket"], ledgers["coin_city"]
    )
    all_training = mechanism_rows + poly_rows + coin_rows
    source_ids = [str(row["job_id"]) for row in all_training]
    locked_rows = ledgers["polymarket"].get("locked_evaluations", [])
    if len(locked_rows) != 2:
        raise AssertionError("expected two Polymarket locked evaluations")
    locked_ids = [str(row["job_id"]) for row in locked_rows]
    if len(set(source_ids + locked_ids)) != 44:
        raise AssertionError("source recovery roster contains duplicate IDs")
    states = scheduler_states(source_ids + locked_ids)
    ineligible = {job_id: states.get(job_id, "UNKNOWN") for job_id in source_ids
                  if states.get(job_id) not in {"PENDING", "RUNNING"}}
    if ineligible:
        raise SystemExit(f"refusing to replace non-live source jobs: {ineligible}")

    mapping: dict[str, dict] = {}
    groups = (("mechanism", mechanism_rows), ("polymarket", poly_rows),
              ("coin_city", coin_rows))
    for campaign, rows in groups:
        for row in rows:
            old = str(row["job_id"])
            command = repaired_command(list(row["command"]), old, walltime=NEW_TIME_LIMIT)
            new = submit(command, dry_run=args.dry_run)
            mapping[old] = {
                "replacement_job_id": new,
                "source_state": states[old],
                "campaign": campaign,
                "cell": {key: row.get(key) for key in
                         ("disclosure", "model", "arm", "seed") if key in row},
                "command": command,
                "scientific_configuration": "identical",
                "scheduler_change_only": {"time_limit": NEW_TIME_LIMIT},
            }

    locked_mapping = {}
    for item in locked_rows:
        model = item["model"]
        model_rows = [row for row in poly_rows if row["model"] == model]
        command, adapter_spec = locked_eval_command(item, model_rows, mapping)
        new = submit(command, dry_run=args.dry_run)
        locked_mapping[str(item["job_id"])] = {
            "replacement_job_id": new, "model": model, "command": command,
            "adapter_spec": adapter_spec,
        }

    if args.dry_run:
        print(f"DRY RUN: would replace {len(mapping)} training and 2 locked-eval jobs")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery_common = {
        "registered_at": timestamp,
        "reason": "20/24-hour allocations were below observed 8B training-plus-endpoint throughput; in-place TimeLimit update was permission-denied",
        "scientific_configuration": "identical",
        "scheduler_change_only": {"time_limit": NEW_TIME_LIMIT},
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
    }

    derived = {name: copy.deepcopy(payload) for name, payload in ledgers.items()}
    replace_rows(derived["mechanism"], [row for row in derived["mechanism"]["submissions"]
                 if row.get("kind") == "training" and row.get("model") in AT_RISK_MODELS], mapping)
    replace_rows(derived["polymarket"], derived["polymarket"]["submissions"], mapping)
    replace_rows(derived["coin_city"], [row for row in derived["coin_city"]["full_training"]
                 if row.get("model") in AT_RISK_MODELS], mapping)
    for item in derived["polymarket"]["locked_evaluations"]:
        old = str(item["job_id"])
        replacement = locked_mapping[old]
        item["job_id"] = replacement["replacement_job_id"]
        item["replaces_job_id"] = old
        item["command"] = replacement["command"]
        item["adapter_spec"] = replacement["adapter_spec"]

    for name in derived:
        relevant = {old: value for old, value in mapping.items()
                    if value["campaign"] == name}
        derived[name]["scheduler_recovery"] = {
            **recovery_common,
            "source_ledger": str(source_paths[name].relative_to(REPO)),
            "source_ledger_sha256": sha256(source_paths[name]),
            "training_replacements": relevant,
        }
    derived["polymarket"]["scheduler_recovery"]["locked_evaluation_replacements"] = locked_mapping

    paths = {
        name: write_derived(source_paths[name], payload, timestamp, "walltime30h")
        for name, payload in derived.items()
    }
    current_ids = [str(row["job_id"]) for row in derived["mechanism"]["submissions"]
                   if row.get("kind") in {"training", "base_evaluation"}]
    current_ids += [str(row["job_id"]) for row in derived["polymarket"]["submissions"]]
    current_ids += [str(row["job_id"]) for row in derived["polymarket"]["locked_evaluations"]]
    coin_ids = [str(row["job_id"]) for row in derived["coin_city"]["full_training"]]
    coin_ids += [str(row["job_id"]) for row in derived["coin_city"]["base_evaluations"]]
    current_finalizer = submit(finalizer_command(
        ROOT / "finalize_current_campaign.sbatch", current_ids,
        {"MECHANISM_LEDGER": str(paths["mechanism"]),
         "POLY_EXTENSION_LEDGER": str(paths["polymarket"])},
    ), dry_run=False)
    coin_finalizer = submit(finalizer_command(
        COIN / "finalize_campaign.sbatch", coin_ids,
        {"CAMPAIGN_LEDGER": str(paths["coin_city"])},
    ), dry_run=False)

    cancel_ids = source_ids + locked_ids + [args.old_current_finalizer, args.old_coin_finalizer]
    subprocess.run(["scancel", *cancel_ids], check=True)

    handoff = {
        **recovery_common,
        "derived_ledgers": {name: str(path.relative_to(REPO)) for name, path in paths.items()},
        "training_replacements": mapping,
        "locked_evaluation_replacements": locked_mapping,
        "cancelled_source_job_ids": source_ids + locked_ids,
        "cancelled_superseded_finalizers": [args.old_current_finalizer, args.old_coin_finalizer],
        "replacement_finalizers": {"current": current_finalizer, "coin_city": coin_finalizer},
    }
    handoff_path = ROOT / "runs" / f"walltime_recovery_{timestamp}.json"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"handoff -> {handoff_path}")
    print(f"current finalizer -> {current_finalizer}")
    print(f"Coin City finalizer -> {coin_finalizer}")


if __name__ == "__main__":
    main()
