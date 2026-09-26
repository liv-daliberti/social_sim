#!/usr/bin/env python3
"""Read-only monitor for the gated Coin City structural-transfer campaign."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
STEP_RE = re.compile(r"['\"]misc/global_step['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")
FAILURES = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}
ACTIVE = {"CONFIGURING", "COMPLETING", "RUNNING", "RESIZING", "SUSPENDED"}


def latest(pattern: str, directory: Path) -> Path:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise SystemExit(f"no file matching {directory / pattern}")
    return paths[-1]


def jobs_from_ledger(path: Path) -> list[dict]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for row in ledger.get("canaries", []):
        jobs.append({**row, "stage": "canary", "kind": "training", "steps": 10})
    gate = ledger.get("canary_gate", {})
    if gate.get("job_id"):
        jobs.append({"job_id": gate["job_id"], "stage": "gate", "kind": "gate"})
    for row in ledger.get("full_training", []):
        jobs.append({**row, "stage": row["kind"], "kind": "training", "steps": 300})
    for row in ledger.get("base_evaluations", []):
        jobs.append({**row, "stage": "base", "kind": "base"})
    expected = Counter(row["stage"] for row in jobs)
    if expected != Counter({"canary": 6, "gate": 1, "confirmatory": 18,
                            "diagnostic": 3, "base": 3}):
        raise SystemExit(f"unexpected Coin City roster: {dict(expected)}")
    ids = [str(row["job_id"]) for row in jobs]
    if len(set(ids)) != 31 or any(not value.isdigit() for value in ids):
        raise SystemExit("Coin City ledger does not contain 31 distinct numeric IDs")
    return jobs


def add_finalizer(jobs: list[dict], handoff: Path | None) -> None:
    if handoff is None:
        candidates = sorted((ROOT.parent / "runs").glob("paper_handoff_*.json"))
        handoff = candidates[-1] if candidates else None
    if handoff is None:
        return
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    job_id = str(
        payload.get("coin_city_structural_campaign", {}).get("finalizer_job_id")
        or payload.get("replacement_finalizers", {}).get("coin_city")
        or ""
    )
    if job_id.isdigit():
        jobs.append({"job_id": job_id, "stage": "paper", "kind": "paper"})
    verifier_id = str((payload.get("paper_verifier") or {}).get("job_id") or "")
    if verifier_id:
        if not verifier_id.isdigit():
            raise SystemExit(f"handoff has nonnumeric paper verifier: {handoff}")
        if any(str(row["job_id"]) == verifier_id for row in jobs):
            raise SystemExit(f"paper verifier duplicates registered job {verifier_id}")
        jobs.append({
            "job_id": verifier_id,
            "stage": "completion-verifier",
            "kind": "verification",
        })


def slurm(ids: list[str]) -> dict[str, dict[str, str]]:
    common = ",".join(ids)
    account = subprocess.run([
        "sacct", "-X", "-n", "-P", "-j", common,
        "-o", "JobIDRaw,State,Elapsed,Timelimit,NodeList,Reason,ExitCode",
    ], text=True, capture_output=True, check=False)
    records = {}
    for line in account.stdout.splitlines():
        fields = line.split("|", 6)
        if len(fields) == 7 and "." not in fields[0]:
            records[fields[0]] = dict(zip(
                ("state", "elapsed", "limit", "node", "reason", "exit"), fields[1:]
            ))
    queue = subprocess.run([
        "squeue", "-h", "-j", common, "-o", "%i|%T|%M|%l|%N|%R"
    ], text=True, capture_output=True, check=False)
    for line in queue.stdout.splitlines():
        fields = line.split("|", 5)
        if len(fields) == 6:
            records[fields[0]] = {
                **records.get(fields[0], {}), "state": fields[1], "elapsed": fields[2],
                "limit": fields[3], "node": fields[4], "reason": fields[5].strip("()"),
            }
    return records


def report_root(job: dict) -> Path | None:
    job_id = str(job["job_id"])
    if job["kind"] == "base":
        paths = list((ROOT / "reports").glob(f"base_{job['model']}_j{job_id}"))
    elif job["kind"] == "training":
        prefix = "canary_" if job["stage"] == "canary" else ""
        paths = list((ROOT / "reports").glob(
            f"{prefix}{job['arm']}_{job['model']}_s{job['seed']}_*_j{job_id}"
        ))
    else:
        paths = []
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def progress(job: dict) -> str:
    maximum = job.get("steps")
    if maximum is None:
        return "-"
    root = report_root(job)
    if root is None:
        return f"0/{maximum}"
    log = root / "train.log"
    if not log.exists():
        return f"0/{maximum}"
    with log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 2 * 1024 * 1024))
        text = handle.read().decode("utf-8", errors="replace").replace("\r", "\n")
    values = [int(float(value)) for value in STEP_RE.findall(text)]
    step = min(max(values, default=0), int(maximum))
    return f"{step}/{maximum} ({100 * step / maximum:.0f}%)"


def label(job: dict) -> str:
    if job["kind"] == "training":
        return f"{job['stage']}/{job['model']}/{job['arm']}/s{job['seed']}"
    if job["kind"] == "base":
        return f"base/{job['model']}"
    return job["stage"]


def render(ledger: Path, jobs: list[dict]) -> tuple[str, bool]:
    status = slurm([str(job["job_id"]) for job in jobs])
    counts = Counter()
    lines = [f"Coin City structural campaign | {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"ledger: {ledger}", "", "JOB       CELL                                           STATE      PROGRESS       NODE / REASON",
             "--------  ---------------------------------------------  ---------  -------------  ----------------"]
    failed = False
    for job in jobs:
        record = status.get(str(job["job_id"]), {})
        state = str(record.get("state", "UNKNOWN")).split()[0].split("+", 1)[0]
        counts[state] += 1
        failed |= state in FAILURES
        place = record.get("node") or record.get("reason") or "-"
        lines.append(f"{job['job_id']:<8}  {label(job):<45}  {state:<9}  "
                     f"{progress(job):<13}  {place}")
    lines.extend(("", "summary: " + ", ".join(
        f"{count} {state.lower()}" for state, count in sorted(counts.items())
    )))
    return "\n".join(lines), failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--watch", nargs="?", const=60.0, type=float)
    args = parser.parse_args()
    ledger = args.ledger or latest("coin_city_structural_*.json", ROOT / "runs")
    jobs = jobs_from_ledger(ledger)
    add_finalizer(jobs, args.handoff)
    while True:
        output, failed = render(ledger, jobs)
        print(output, flush=True)
        if args.watch is None:
            return 1 if failed else 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
