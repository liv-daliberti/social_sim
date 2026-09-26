#!/usr/bin/env python3
"""Monitor the current structural-OOD and Polymarket-extension campaign.

The newest timestamped ledger for each experiment is selected by default. The
monitor is read-only: it queries Slurm, discovers logs and reports, and renders
locked-test metrics once they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MECHANISM = ROOT / "mechanism_family"
POLY = ROOT / "polymarket"
FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE_STATES = {"CONFIGURING", "COMPLETING", "RUNNING", "RESIZING", "SUSPENDED"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"['\"]misc/global_step['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")
TQDM_RE = re.compile(r"\b([0-9]+)/([0-9]+)\s*\[")


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    experiment: str
    role: str
    seed: int | None
    ledger: Path
    max_steps: int | None
    dependencies: tuple[str, ...] = ()
    model: str | None = None
    disclosure: str | None = None
    arm: str | None = None

    @property
    def label(self) -> str:
        seed = f" s{self.seed}" if self.seed is not None else ""
        return f"{self.experiment} {self.role}{seed}"


class MonitorError(RuntimeError):
    pass


def latest_ledger(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise MonitorError(f"no ledger matching {directory / pattern}")
    return paths[-1]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot read ledger {path}: {exc}") from exc


def clean_job_id(value: Any) -> str:
    return str(value or "").split(";", 1)[0].strip()


def registered_steps(environment: dict[str, Any]) -> int | None:
    try:
        presentations = int(environment["MAX_TRAIN"])
        batch = int(environment["BATCH"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return presentations // batch if batch > 0 else None


def latest_full_mechanism_ledger() -> Path:
    for path in sorted((MECHANISM / "runs").glob("c3_mechanism_*.json"), reverse=True):
        payload = read_json(path)
        training = [
            item
            for item in payload.get("submissions", [])
            if item.get("kind") == "training"
        ]
        bases = [
            item
            for item in payload.get("submissions", [])
            if item.get("kind") == "base_evaluation"
        ]
        if len(training) == 42 and len(bases) == 6:
            return path
    raise MonitorError("no exact 42-training/6-base mechanism ledger found")


def discover_expanded_jobs(
    mechanism_ledger: Path | None = None,
    polymarket_extension_ledger: Path | None = None,
) -> tuple[list[JobSpec], dict[str, str]]:
    mechanism_path = mechanism_ledger or latest_full_mechanism_ledger()
    extension_path = polymarket_extension_ledger or latest_ledger(
        POLY / "runs", "exp3b_model_extension_*.json"
    )
    mechanism = read_json(mechanism_path)
    extension = read_json(extension_path)
    jobs: list[JobSpec] = []

    for item in mechanism.get("submissions", []):
        job_id = clean_job_id(item.get("job_id"))
        kind = str(item.get("kind"))
        if not job_id or kind not in {"training", "base_evaluation"}:
            continue
        environment = item.get("environment") or {}
        arm = str(item.get("arm") or "base")
        jobs.append(
            JobSpec(
                job_id=job_id,
                experiment="C3",
                role=f"{item.get('disclosure')}/{arm}/{item.get('model')}",
                seed=item.get("seed") if kind == "training" else None,
                ledger=mechanism_path,
                max_steps=registered_steps(environment) if kind == "training" else None,
                model=str(item.get("model")),
                disclosure=str(item.get("disclosure")),
                arm=arm,
            )
        )

    for item in extension.get("submissions", []):
        job_id = clean_job_id(item.get("job_id"))
        if not job_id:
            continue
        environment = item.get("environment") or {}
        jobs.append(
            JobSpec(
                job_id=job_id,
                experiment="3B-ext",
                role=f"train/{item.get('model')}",
                seed=item.get("seed"),
                ledger=extension_path,
                max_steps=registered_steps(environment) or 300,
                model=str(item.get("model")),
                arm="train",
            )
        )
    for item in extension.get("locked_evaluations", []):
        job_id = clean_job_id(item.get("job_id"))
        if not job_id:
            continue
        dependencies = tuple(
            value.split(":", 1)[1]
            for value in str(item.get("adapter_spec", "")).split(";")
            if ":" in value
        )
        jobs.append(
            JobSpec(
                job_id=job_id,
                experiment="3B-ext",
                role=f"locked_test/{item.get('model')}",
                seed=None,
                ledger=extension_path,
                max_steps=None,
                dependencies=dependencies,
                model=str(item.get("model")),
                arm="locked_test",
            )
        )

    if len([job for job in jobs if job.experiment == "C3"]) != 48:
        raise MonitorError("expanded mechanism ledger does not resolve to 48 jobs")
    if len([job for job in jobs if job.experiment == "3B-ext"]) != 8:
        raise MonitorError("Polymarket extension ledger does not resolve to 8 jobs")
    if len({job.job_id for job in jobs}) != 56:
        raise MonitorError("expanded campaign does not contain 56 distinct job IDs")
    return jobs, {"C3": str(mechanism_path), "3B-ext": str(extension_path)}


def add_current_finalizer(
    jobs: list[JobSpec], ledgers: dict[str, str], handoff_path: Path | None
) -> None:
    """Add the registered broad-campaign finalizer and cross-paper verifier."""
    if handoff_path is None:
        return
    path = handoff_path.resolve()
    handoff = read_json(path)
    job_id = clean_job_id(
        (handoff.get("replacement_finalizers") or {}).get("current")
        or (handoff.get("current_campaign") or {}).get("finalizer_job_id")
    )
    if not job_id.isdigit():
        raise MonitorError(f"handoff has no numeric current finalizer: {path}")
    if any(job.job_id == job_id for job in jobs):
        raise MonitorError(f"current finalizer duplicates scientific job {job_id}")
    jobs.append(
        JobSpec(
            job_id=job_id,
            experiment="paper",
            role="current-finalizer",
            seed=None,
            ledger=path,
            max_steps=None,
            dependencies=tuple(job.job_id for job in jobs),
            arm="paper",
        )
    )
    verifier_id = clean_job_id((handoff.get("paper_verifier") or {}).get("job_id"))
    if verifier_id:
        if not verifier_id.isdigit():
            raise MonitorError(f"handoff has nonnumeric paper verifier: {path}")
        if any(job.job_id == verifier_id for job in jobs):
            raise MonitorError(
                f"paper verifier duplicates registered job {verifier_id}"
            )
        verifier_dependencies = tuple(
            clean_job_id(value)
            for value in (handoff.get("paper_verifier") or {}).get(
                "dependency_job_ids", []
            )
            if clean_job_id(value)
        )
        jobs.append(
            JobSpec(
                job_id=verifier_id,
                experiment="paper",
                role="completion-verifier",
                seed=None,
                ledger=path,
                max_steps=None,
                dependencies=verifier_dependencies,
                arm="verification",
            )
        )
    ledgers["handoff"] = str(path)


def run_command(command: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "", f"command not found: {command[0]}"
    if result.returncode:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        return "", f"{command[0]} failed: {detail}"
    return result.stdout, None


def parse_squeue(text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        fields = line.split("|", 4)
        if len(fields) != 5:
            continue
        job_id, state, elapsed, limit, reason = fields
        records[job_id] = {
            "state": state,
            "elapsed": elapsed,
            "limit": limit,
            "reason": reason.strip("()"),
            "source": "squeue",
        }
    return records


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        fields = line.split("|", 9)
        if len(fields) != 10:
            continue
        (
            job_id,
            name,
            state,
            elapsed,
            limit,
            start,
            end,
            exit_code,
            node,
            reason,
        ) = fields
        if "." in job_id:
            continue
        records[job_id] = {
            "name": name,
            "state": state,
            "elapsed": elapsed,
            "limit": limit,
            "start": start,
            "end": end,
            "exit_code": exit_code,
            "reason": reason if reason not in {"", "None"} else node,
            "source": "sacct",
        }
    return records


def slurm_status(job_ids: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    ids = ",".join(job_ids)
    warnings: list[str] = []
    queue_text, error = run_command(["squeue", "-h", "-j", ids, "-o", "%i|%T|%M|%l|%R"])
    if error:
        warnings.append(error)
    account_text, error = run_command(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            ids,
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,ExitCode,NodeList,Reason",
        ]
    )
    if error:
        warnings.append(error)
    records = parse_sacct(account_text)
    for job_id, queued in parse_squeue(queue_text).items():
        records[job_id] = {**records.get(job_id, {}), **queued}
    return records, warnings


def normalized_state(value: str) -> str:
    return value.upper().split()[0].split("+", 1)[0] if value else "UNKNOWN"


def tail_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            raw = handle.read()
    except OSError:
        return ""
    return ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).replace("\r", "\n")


def newest(paths: list[Path]) -> Path | None:
    return (
        max(paths, key=lambda path: (path.stat().st_mtime, str(path)))
        if paths
        else None
    )


def report_directory(job: JobSpec) -> Path | None:
    if job.experiment == "C3":
        if job.arm == "base":
            pattern = f"base_{job.disclosure}_{job.model}_j{job.job_id}"
        else:
            pattern = (
                f"{job.disclosure}_{job.arm}_{job.model}_"
                f"s{job.seed}_*_j{job.job_id}"
            )
        return newest(list((MECHANISM / "reports").glob(pattern)))
    if job.experiment == "3B-ext":
        if job.arm == "locked_test":
            return None
        return newest(
            list((POLY / "reports").glob(f"train_{job.model}_market_*_j{job.job_id}"))
        )
    return None


def log_paths(job: JobSpec, report: Path | None) -> list[Path]:
    if job.experiment == "paper" and job.role == "completion-verifier":
        slurm_base = ROOT / "logs" / f"verify_paper_{job.job_id}"
    elif job.experiment == "paper":
        slurm_base = ROOT / "logs" / f"finalize_current_{job.job_id}"
    elif job.experiment == "C3":
        prefix = "base" if job.arm == "base" else "train"
        slurm_base = MECHANISM / "logs" / f"{prefix}_{job.job_id}"
    elif job.experiment == "3B-ext":
        prefix = "ext_test" if job.arm == "locked_test" else "ext"
        slurm_base = POLY / "logs" / f"{prefix}_{job.job_id}"
    else:
        raise MonitorError(f"unsupported current campaign job: {job.label}")
    paths = [slurm_base.with_suffix(".out"), slurm_base.with_suffix(".err")]
    if report is not None:
        paths.insert(0, report / "train.log")
    return [path for path in paths if path.exists()]


def artifact_status(job: JobSpec) -> dict[str, Any]:
    report = report_directory(job)
    result: dict[str, Any] = {
        "report": str(report) if report else None,
        "logs": [str(path) for path in log_paths(job, report)],
        "step": None,
        "max_steps": job.max_steps,
        "latest_eval_step": None,
        "latest_checkpoint": None,
    }
    if job.experiment == "3B-ext" and job.arm == "locked_test":
        prefix = POLY / "reports" / f"exp3b_{job.model}_locked_test_j{job.job_id}"
        summary = prefix.with_suffix(".summary.json")
        predictions = prefix.with_suffix(".jsonl")
        result["locked_summary"] = str(summary) if summary.exists() else None
        result["locked_predictions"] = (
            str(predictions) if predictions.exists() else None
        )
        return result
    if report is None:
        return result

    train_log = report / "train.log"
    text = tail_text(train_log) if train_log.exists() else ""
    if train_log.exists():
        result["log_mtime"] = train_log.stat().st_mtime
        result["log_bytes"] = train_log.stat().st_size
    steps = [int(float(value)) for value in STEP_RE.findall(text)]
    result["train_steps_seen"] = len(steps)
    # 2026-08-11: the tqdm fallback used to accept ANY "<done>/<total> [" bar, which matched the
    # dataset-map progress bar ("4800/4800 ["). progress_text then clamped 4800 to max_steps and
    # rendered "300/300 (100%)" for six jobs that had not taken a single training step -- the whole
    # campaign read as complete while it was deadlocked. Only trust a bar whose TOTAL is the
    # registered step count.
    tqdm = [(int(done), int(total)) for done, total in TQDM_RE.findall(text)]
    tqdm = [
        (d, t)
        for d, t in tqdm
        if result["max_steps"] is not None and t == result["max_steps"]
    ]
    if tqdm:
        steps.append(tqdm[-1][0])

    eval_steps = []
    for path in report.glob("debug_*/eval_results/*.json"):
        try:
            eval_steps.append(int(path.stem))
        except ValueError:
            continue
    registered_evals = [
        step for step in eval_steps if job.max_steps is None or step <= job.max_steps
    ]
    if registered_evals:
        result["latest_eval_step"] = max(registered_evals)
        steps.append(max(registered_evals))

    checkpoints: list[tuple[int, Path]] = []
    for path in report.glob("debug_*/saved_models/step_*"):
        try:
            checkpoints.append((int(path.name.rsplit("_", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    if checkpoints:
        _, checkpoint = max(checkpoints)
        result["latest_checkpoint"] = str(checkpoint)
    if steps:
        result["step"] = max(steps)
    return result


STALL_SECONDS = 20 * 60


def progress_text(artifact: dict[str, Any], active: bool = False) -> str:
    """Render progress, and say STALLED rather than "-" when an ACTIVE job has taken no training
    step and stopped writing its log. A job deadlocked in model init looks identical to one that is
    merely slow to start; the difference is whether the log is still moving."""
    step = artifact.get("step")
    maximum = artifact.get("max_steps")
    if active and not artifact.get("train_steps_seen"):
        mtime = artifact.get("log_mtime")
        if mtime is not None:
            idle = time.time() - float(mtime)
            if idle > STALL_SECONDS:
                return f"STALLED 0 steps, log idle {idle / 3600:.1f}h"
        return "starting (0 steps)"
    if step is None:
        return "-"
    if maximum:
        bounded = min(int(step), int(maximum))
        return f"{bounded}/{maximum} ({100.0 * bounded / maximum:.0f}%)"
    return f"step {step}"


def clipped(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: width - 1] + "~"


def render_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    widths[-1] = min(widths[-1], 30)
    rendered = []
    for number, row in enumerate(rows):
        values = [clipped(value, widths[index]) for index, value in enumerate(row)]
        rendered.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))
        )
        if number == 0:
            rendered.append("  ".join("-" * width for width in widths))
    return "\n".join(rendered)


def frozen_baseline_lines() -> list[str]:
    path = POLY / "reports" / "exp3b_baselines.json"
    if not path.exists():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        test = report["splits"]["test"]
        market = test["market"]["brier"]
        platt = test["platt_market_train_only"]["brier"]
        delta = test["platt_minus_market_brier"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return [f"Frozen baselines: unreadable report {path}"]
    return [
        f"Frozen 3B test baselines: market Brier {market:.5f}; train-only Platt {platt:.5f}",
        f"  Platt - market: {delta['estimate']:+.5f} "
        f"(family-bootstrap 95% CI [{delta['ci95_low']:+.5f}, {delta['ci95_high']:+.5f}])",
    ]


def locked_result_lines(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    summaries = sorted(
        {
            Path(item["locked_summary"])
            for item in artifacts.values()
            if item.get("locked_summary")
        }
    )
    if not summaries:
        return ["Locked 3B learned-model result: pending"]
    lines = []
    for path in summaries:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"Locked 3B result unreadable: {path} ({exc})")
            continue
        model = summary.get("model_key") or summary.get("model") or path.stem
        lines.append(f"Locked 3B result ({model}): {path}")
        for name, metrics in summary.get("models", {}).items():
            lines.append(
                f"  {name}: Brier {metrics['brier']:.5f}; "
                f"parse coverage {metrics['parse_coverage']:.1%}"
            )
        for name, comparison in summary.get("comparisons_brier", {}).items():
            lines.append(
                f"  {name}: {comparison['estimate']:+.5f} "
                f"[{comparison['ci95_low']:+.5f}, {comparison['ci95_high']:+.5f}]"
            )
    return lines


def last_lines(path: Path, count: int) -> list[str]:
    lines = [
        line
        for line in tail_text(path, max_bytes=256 * 1024).splitlines()
        if line.strip()
    ]
    return lines[-count:]


def snapshot(jobs: list[JobSpec], ledgers: dict[str, str]) -> dict[str, Any]:
    slurm, warnings = slurm_status([job.job_id for job in jobs])
    artifacts = {job.job_id: artifact_status(job) for job in jobs}
    rendered_jobs = []
    for job in jobs:
        record = slurm.get(job.job_id, {})
        rendered_jobs.append(
            {
                "job_id": job.job_id,
                "label": job.label,
                "experiment": job.experiment,
                "role": job.role,
                "seed": job.seed,
                "dependencies": list(job.dependencies),
                "state": normalized_state(record.get("state", "")),
                "elapsed": record.get("elapsed", "-"),
                "limit": record.get("limit", "-"),
                "reason": record.get("reason", "not visible in squeue/sacct"),
                "exit_code": record.get("exit_code"),
                "artifact": artifacts[job.job_id],
            }
        )
    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ledgers": ledgers,
        "jobs": rendered_jobs,
        "warnings": warnings,
    }


def render(snapshot_data: dict[str, Any], tail: int = 0) -> str:
    jobs = snapshot_data["jobs"]
    lines = [f"Experiment 3 campaign | {snapshot_data['checked_at']}"]
    for name, path in snapshot_data["ledgers"].items():
        lines.append(f"{name} ledger: {path}")
    lines.append("")
    table = [["JOB", "RUN", "STATE", "ELAPSED", "PROGRESS", "NODE / REASON"]]
    for job in jobs:
        table.append(
            [
                job["job_id"],
                job["label"],
                job["state"],
                f"{job['elapsed']}/{job['limit']}",
                progress_text(job["artifact"], job["state"] in ACTIVE_STATES),
                job["reason"],
            ]
        )
    lines.append(render_table(table))

    states = [job["state"] for job in jobs]
    pending = sum(state == "PENDING" for state in states)
    active = sum(state in ACTIVE_STATES for state in states)
    complete = sum(state == "COMPLETED" for state in states)
    failed = sum(state in FAILURE_STATES for state in states)
    unknown = sum(state == "UNKNOWN" for state in states)
    lines.extend(
        [
            "",
            f"Campaign summary: {pending} pending, {active} active, {complete} complete, "
            f"{failed} failed, {unknown} unknown",
        ]
    )
    lines.extend(frozen_baseline_lines())
    artifacts = {job["job_id"]: job["artifact"] for job in jobs}
    lines.extend(locked_result_lines(artifacts))

    existing = [
        job
        for job in jobs
        if job["artifact"].get("report") or job["artifact"].get("locked_summary")
    ]
    if existing:
        lines.append("")
        lines.append("Artifacts:")
        for job in existing:
            artifact = job["artifact"]
            target = artifact.get("locked_summary") or artifact.get("report")
            lines.append(f"  {job['job_id']} {job['label']}: {target}")
            if artifact.get("latest_checkpoint"):
                lines.append(f"    checkpoint: {artifact['latest_checkpoint']}")

    for warning in snapshot_data["warnings"]:
        lines.append(f"WARNING: {warning}")

    failed_jobs = [job for job in jobs if job["state"] in FAILURE_STATES]
    if failed_jobs:
        lines.append("")
        lines.append("Failures:")
        for job in failed_jobs:
            lines.append(
                f"  {job['job_id']} {job['label']}: {job['state']} "
                f"exit={job.get('exit_code') or '?'} reason={job['reason']}"
            )

    if tail > 0:
        lines.append("")
        lines.append(f"Recent logs (last {tail} nonblank lines):")
        for job in jobs:
            paths = [Path(path) for path in job["artifact"].get("logs", [])]
            if not paths:
                continue
            preferred = paths[0]
            lines.append(f"--- {job['job_id']} {job['label']} | {preferred}")
            lines.extend(f"    {line}" for line in last_lines(preferred, tail))
            error_paths = [
                path for path in paths if path.suffix == ".err" and path.stat().st_size
            ]
            for error_path in error_paths:
                lines.append(f"--- stderr | {error_path}")
                lines.extend(f"    {line}" for line in last_lines(error_path, tail))
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Monitor the current structural-OOD and Polymarket-extension ledgers."
    )
    result.add_argument(
        "--mechanism-ledger",
        type=Path,
        help="override the expanded-C3 mechanism ledger",
    )
    result.add_argument(
        "--polymarket-extension-ledger",
        type=Path,
        help="override the expanded-C3 Polymarket extension ledger",
    )
    result.add_argument(
        "--handoff",
        type=Path,
        help="add the current-paper finalizer registered in this handoff",
    )
    result.add_argument(
        "--watch",
        nargs="?",
        const=30.0,
        type=float,
        metavar="SECONDS",
        help="refresh continuously (default interval: 30 seconds)",
    )
    result.add_argument(
        "--tail",
        type=int,
        default=0,
        metavar="N",
        help="show N recent log lines per job",
    )
    result.add_argument(
        "--json", action="store_true", help="emit a machine-readable snapshot"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.tail < 0:
        raise SystemExit("--tail must be nonnegative")
    if args.watch is not None and args.watch <= 0:
        raise SystemExit("--watch interval must be positive")
    try:
        jobs, ledgers = discover_expanded_jobs(
            args.mechanism_ledger, args.polymarket_extension_ledger
        )
        add_current_finalizer(jobs, ledgers, args.handoff)
    except MonitorError as exc:
        print(f"monitor error: {exc}", file=sys.stderr)
        return 2

    while True:
        data = snapshot(jobs, ledgers)
        output = (
            json.dumps(data, indent=2, sort_keys=True)
            if args.json
            else render(data, args.tail)
        )
        if args.watch is not None and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if args.watch is None:
            return (
                1 if any(job["state"] in FAILURE_STATES for job in data["jobs"]) else 0
            )
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
