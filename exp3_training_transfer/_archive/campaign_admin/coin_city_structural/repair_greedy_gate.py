#!/usr/bin/env python3
"""Submit matched constrained-greedy canary endpoints and rewire the held audit gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(command: list[str], dry_run: bool) -> str | None:
    print(shlex.join(command))
    if dry_run:
        return None
    return subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]


def state_map(job_ids: list[str]) -> dict[str, str]:
    output = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
        "--format=JobIDRaw,State",
    ], text=True)
    states = {}
    for line in output.splitlines():
        if "|" in line:
            job_id, state = line.split("|", 1)
            states[job_id] = state.split()[0].split("+")[0]
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--gate-job-id", required=True)
    parser.add_argument("--partition", default="all")
    parser.add_argument("--account", default="mltheory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gate_job_id.isdigit():
        parser.error("--gate-job-id must be numeric")

    original = json.loads(args.ledger.read_text(encoding="utf-8"))
    if str(original.get("canary_gate", {}).get("job_id")) != args.gate_job_id:
        raise SystemExit("gate job does not match the registered campaign ledger")
    canaries = original.get("canaries", [])
    if len(canaries) != 6 or any(not str(row.get("job_id", "")).isdigit() for row in canaries):
        raise SystemExit("registered campaign does not contain exactly six submitted canaries")
    states = state_map([str(row["job_id"]) for row in canaries])
    allowed = {"COMPLETED", "RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
    invalid = {str(row["job_id"]): states.get(str(row["job_id"]), "UNKNOWN")
               for row in canaries if states.get(str(row["job_id"])) not in allowed}
    if invalid:
        raise SystemExit(f"canaries are not eligible for endpoint repair: {invalid}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    correction_path = ROOT / "runs" / f"coin_city_greedy_correction_{timestamp}.json"
    endpoint_jobs = []
    for row in canaries:
        environment = row["environment"]
        exported = {
            "CANARY_JOB_ID": str(row["job_id"]),
            "ARM": str(row["arm"]),
            "MODEL_KEY": str(row["model"]),
            "MODEL": str(environment["MODEL"]),
            "PROMPT_TEMPLATE": str(environment["PROMPT_TEMPLATE"]),
            "SEED": str(row["seed"]),
        }
        export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exported.items())
        command = [
            "sbatch", "--parsable", f"--partition={args.partition}",
            f"--account={args.account}", "--cpus-per-task=8", "--mem=24G",
            "--time=01:00:00", "--gres=gpu:a6000:1", "--exclude=node206",
        ]
        if states[str(row["job_id"])] != "COMPLETED":
            command.append(f"--dependency=afterok:{row['job_id']}")
        command.extend([f"--export={export_arg}", str(ROOT / "greedy_endpoint.sh")])
        job_id = submit(command, args.dry_run)
        endpoint_jobs.append({
            "model": row["model"], "arm": row["arm"], "seed": row["seed"],
            "canary_job_id": row["job_id"], "job_id": job_id,
            "environment": exported, "command": command,
        })

    endpoint_ids = [str(row["job_id"]) for row in endpoint_jobs if row["job_id"]]
    correction = {
        "protocol": "coin_city_constrained_greedy_correction_v1",
        "created_at_utc": timestamp,
        "original_ledger": str(args.ledger),
        "original_ledger_sha256": sha256(args.ledger),
        "reason": (
            "Replace the training evaluator's unconstrained greedy dump with a matched "
            "xgrammar-constrained temperature-zero endpoint for every canary and future run."
        ),
        "invariants": [
            "saved adapters unchanged", "training rewards unchanged",
            "held-out dataset unchanged", "stochastic endpoints unchanged",
            "greedy temperature remains zero", "one greedy draw per task",
        ],
        "source_sha256": {path.name: sha256(path) for path in (
            ROOT / "train.sh", ROOT / "greedy_endpoint.sh", ROOT / "canary_audit.py",
            ROOT / "make_paper_outputs.py", ROOT.parent / "mechanism_family" / "evaluate_endpoint.py",
            ROOT / "report.py",
        )},
        "endpoint_jobs": endpoint_jobs,
        "gate_job_id": args.gate_job_id,
        "gate_dependency": ("afterok:" + ":".join(endpoint_ids)) if endpoint_ids else None,
        "downstream_dependency": f"afterok:{args.gate_job_id}",
        "status": "submitted" if not args.dry_run else "dry_run",
    }
    correction_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")

    if not args.dry_run:
        if len(endpoint_ids) != 6:
            raise SystemExit("not all six endpoint jobs were submitted; gate remains held")
        subprocess.run([
            "scontrol", "update", f"JobId={args.gate_job_id}",
            "Dependency=afterok:" + ":".join(endpoint_ids),
        ], cwd=REPO, check=True)
        subprocess.run(["scontrol", "release", args.gate_job_id], cwd=REPO, check=True)
        correction["status"] = "rewired"
        correction_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    print(f"correction ledger -> {correction_path}")


if __name__ == "__main__":
    main()
