#!/usr/bin/env python3
"""CPU-only finalization of the fixed three-model development pilot.

Intended for one Slurm afterany job. Queries scheduler accounting once, preserves
raw responses, analyzes every prescribed model, and aggregates only a complete
set of analysis summaries. Missing raw files are represented by explicitly empty
files in the run's missing_responses directory so planned denominators survive.
No model calls, GPU imports, retries, submissions, or performance-based selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

MODEL_KEYS = ("qwen2_5_72b_instruct", "llama3_1_70b_instruct", "qwen3_32b")
EXPECTED_RECORDS = 960
DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs/development_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def query_scheduler(job_ids: list[str]) -> dict[str, Any]:
    command = ["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids),
               "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End"]
    result = run_command(command, timeout=30)
    states = {job_id: {"job_id": job_id, "state": "UNKNOWN", "accounting_row_present": False}
              for job_id in job_ids}
    if result["returncode"] == 0:
        for line in result["stdout"].splitlines():
            fields = line.rstrip("|").split("|")
            if len(fields) < 6 or fields[0] not in states:
                continue
            job_id, raw_state, exit_code, elapsed, started, ended = fields[:6]
            # E.g. CANCELLED by 12345 and truncated states remain non-complete.
            state = raw_state.strip().split(" ", 1)[0].rstrip("+")
            states[job_id] = {
                "job_id": job_id, "state": state, "raw_state": raw_state,
                "exit_code": exit_code, "elapsed": elapsed, "started_at": started,
                "ended_at": ended, "accounting_row_present": True,
            }
    return {"queried_at": utc_now(), "query": result, "jobs": states}


def manifest_check(manifest: dict, job: dict, design_hash: str, runner_hash: str) -> dict[str, Any]:
    config = manifest.get("config", {})
    checks = {
        "status_complete": manifest.get("status") == "complete",
        "planned_records_match": manifest.get("expected_records") == EXPECTED_RECORDS,
        "record_count_complete": manifest.get("records") == EXPECTED_RECORDS,
        "all_statuses_ok": manifest.get("counts") == {"ok": EXPECTED_RECORDS},
        "model_key_matches": config.get("model_key") == job["model_key"],
        "model_path_matches": config.get("local_model", {}).get("path") == job["model_path"],
        "design_hash_matches": config.get("input_sha256") == design_hash,
        "runner_hash_matches": config.get("runner_sha256") == runner_hash,
        "job_id_recorded": str(job["job_id"]) in manifest.get("slurm_job_ids", []),
    }
    return {"complete_and_consistent": all(checks.values()), "checks": checks}


def coverage_complete(coverage: dict) -> bool:
    return (coverage.get("expected") == EXPECTED_RECORDS
            and coverage.get("present") == EXPECTED_RECORDS
            and coverage.get("valid") == EXPECTED_RECORDS
            and coverage.get("missing_record") == 0
            and coverage.get("invalid_present") == 0)


def finalize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    completion_path = run_dir / "completion.json"
    completion: dict[str, Any] = {
        "schema_version": "context_reversal_run_completion_v1", "run_dir": str(run_dir),
        "started_at": utc_now(), "status": "finalizing", "plan_complete": False,
        "material_status": "authored_development_unvalidated",
        "independent_human_validation": "not_collected", "frontier_api_calls": 0,
        "prescribed_models": list(MODEL_KEYS), "expected_records_per_model": EXPECTED_RECORDS,
        "models": {}, "errors": [],
    }
    write_json(completion_path, completion)
    try:
        submission_path = run_dir / "submission.json"
        submission = json.loads(submission_path.read_text())
        completion["submission_sha256"] = sha256(submission_path)
        jobs = submission["jobs"]
        if (len(jobs) != 3 or {j["model_key"] for j in jobs} != set(MODEL_KEYS)
                or len({str(j["job_id"]) for j in jobs}) != 3):
            raise ValueError("submission.jobs must contain exactly the three prescribed models and distinct job IDs")
        if (submission.get("planned_calls_per_model") != EXPECTED_RECORDS
                or submission.get("planned_total_calls") != EXPECTED_RECORDS * 3):
            raise ValueError("Submission does not describe the fixed 960-record-per-model plan")
        jobs_by_model = {job["model_key"]: job for job in jobs}
        results_roots = {Path(job["results_dir"]).resolve().parent for job in jobs}
        if len(results_roots) != 1:
            raise ValueError("All prescribed results directories must share the same results root")
        results_root = results_roots.pop()
        if any(Path(job["results_dir"]).resolve().name != job["model_key"] for job in jobs):
            raise ValueError("Per-model results directory does not match the model key")
        design = run_dir / "development_plan.jsonl"
        analyzer = run_dir / "code/analyze.py"
        aggregator = run_dir / "code/aggregate.py"
        design_hash = sha256(design)
        if design_hash != submission["design_sha256"]:
            raise ValueError("Frozen design SHA-256 differs from submission")
        if sha256(analyzer) != submission["files_sha256"]["analyze.py"]:
            raise ValueError("Frozen analyzer SHA-256 differs from submission")
        runner_hash = submission["files_sha256"]["run_local.py"]
        completion["design_sha256"] = design_hash
        completion["results_root"] = str(results_root)
        completion["scheduler"] = query_scheduler([str(jobs_by_model[key]["job_id"]) for key in MODEL_KEYS])
        for key in MODEL_KEYS:
            job = jobs_by_model[key]
            response = Path(job["response_path"]).resolve()
            manifest_source = Path(str(response) + ".manifest.json")
            model: dict[str, Any] = {
                "model_key": key, "job_id": str(job["job_id"]),
                "scheduler": completion["scheduler"]["jobs"][str(job["job_id"])],
                "raw_response_path": str(response), "raw_response_present": response.is_file(),
                "manifest_source": str(manifest_source), "manifest_present": manifest_source.is_file(),
                "manifest_complete": False, "analysis_succeeded": False, "plan_complete": False,
            }
            completion["models"][key] = model
            try:
                if manifest_source.is_file():
                    payload = manifest_source.read_bytes()
                    manifest = json.loads(payload)
                    if not isinstance(manifest, dict):
                        raise ValueError("Response manifest is not a JSON object")
                    copied = run_dir / f"{key}.manifest.json"
                    atomic_bytes(copied, payload)
                    model["manifest_copy"] = str(copied)
                    model["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
                    model["manifest_status"] = manifest.get("status")
                    model["manifest_counts"] = manifest.get("counts")
                    checked = manifest_check(manifest, job, design_hash, runner_hash)
                    model["manifest_checks"] = checked["checks"]
                    model["manifest_complete"] = checked["complete_and_consistent"]
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                model["manifest_error"] = str(exc)
            try:
                if response.is_file():
                    analysis_input = response
                    model["raw_response_sha256_before_analysis"] = sha256(response)
                else:
                    analysis_input = run_dir / "missing_responses" / f"{key}.jsonl"
                    analysis_input.parent.mkdir(parents=True, exist_ok=True)
                    if analysis_input.exists() and analysis_input.read_bytes():
                        raise ValueError("Missing-response placeholder already contains data; refusing to overwrite it")
                    if not analysis_input.exists():
                        analysis_input.write_bytes(b"")
                    model["missing_input_representation"] = "explicit_empty_file_preserving_all_planned_denominators"
                model["analysis_input"] = str(analysis_input)
                command = [sys.executable, str(analyzer), "--design", str(design),
                           "--responses", str(analysis_input), "--model-key", key,
                           "--output", str(Path(job["results_dir"]).resolve())]
                print(f"Analyzing {key}: raw_response_present={model['raw_response_present']}", flush=True)
                result = run_command(command, timeout=600)
                model["analysis_process"] = result
                if response.is_file():
                    model["raw_response_sha256_after_analysis"] = sha256(response)
                    model["raw_response_unchanged"] = (model.get("raw_response_sha256_before_analysis")
                                                        == model["raw_response_sha256_after_analysis"])
                if result["returncode"] != 0:
                    raise ValueError(f"Frozen analyzer failed for {key}; see analysis_process")
                summary_path = Path(job["results_dir"]).resolve() / "summary.json"
                summary = json.loads(summary_path.read_text())
                if (summary.get("model_key") != key
                        or summary.get("provenance", {}).get("design", {}).get("sha256") != design_hash):
                    raise ValueError("Analysis summary model/design provenance mismatch")
                coverage = summary["coverage"]["totals"]
                model["analysis_succeeded"] = True
                model["summary_path"] = str(summary_path)
                model["summary_sha256"] = sha256(summary_path)
                model["coverage"] = coverage
                model["coverage_complete"] = coverage_complete(coverage)
                model["plan_complete"] = bool(
                    model["scheduler"]["state"] == "COMPLETED"
                    and model["manifest_complete"] and model["coverage_complete"]
                    and model["raw_response_present"] and model.get("raw_response_unchanged")
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                model["analysis_error"] = str(exc)
            write_json(completion_path, completion)

        # Do not aggregate a selected subset or re-use a stale summary after a failed analysis.
        if all(completion["models"][key]["analysis_succeeded"] for key in MODEL_KEYS):
            prefix = results_root / "development_summary"
            result = run_command([sys.executable, str(aggregator), "--results-root", str(results_root),
                                  "--output-prefix", str(prefix)], timeout=120)
            completion["aggregate_process"] = result
            completion["aggregate_succeeded"] = result["returncode"] == 0
            if result["returncode"] == 0:
                completion["aggregate_json"] = str(prefix.with_suffix(".json"))
                completion["aggregate_markdown"] = str(prefix.with_suffix(".md"))
        else:
            completion["aggregate_succeeded"] = False
            completion["aggregate_skipped_reason"] = "One or more prescribed-model analyses failed; no subset aggregate produced"
        completion["plan_complete"] = bool(
            completion.get("aggregate_succeeded")
            and all(completion["models"][key]["plan_complete"] for key in MODEL_KEYS)
        )
        completion["coverage_totals"] = {
            "expected": EXPECTED_RECORDS * len(MODEL_KEYS),
            "valid": sum(item.get("coverage", {}).get("valid", 0) for item in completion["models"].values()),
            "models_with_coverage": sum("coverage" in item for item in completion["models"].values()),
        }
        if completion.get("aggregate_succeeded"):
            prefix = results_root / "development_summary"
            report_path = prefix.with_suffix(".json")
            report = json.loads(report_path.read_text())
            status = {
                "plan_complete": completion["plan_complete"], "completion_manifest": str(completion_path),
                "coverage_totals": completion["coverage_totals"],
                "models": {key: {"job_id": item["job_id"], "scheduler_state": item["scheduler"]["state"],
                                 "manifest_complete": item["manifest_complete"], "coverage": item["coverage"],
                                 "plan_complete": item["plan_complete"]}
                           for key, item in completion["models"].items()},
            }
            report["run_completion"] = status
            write_json(report_path, report)
            markdown_path = prefix.with_suffix(".md")
            lines = ["", "## Scheduler and run completeness", "",
                     f"Plan complete: **{str(completion['plan_complete']).lower()}**. See `{completion_path}` for accounting and provenance.", "",
                     "| Model | Slurm job | Scheduler state | Complete manifest | Valid / planned records |",
                     "|---|---|---|---|---:|"]
            for key, item in completion["models"].items():
                counts = item["coverage"]
                lines.append(f"| {key} | {item['job_id']} | {item['scheduler']['state']} | {str(item['manifest_complete']).lower()} | {counts['valid']} / {counts['expected']} |")
            lines.extend(["", "Missing or failed model runs remain in the prescribed-model aggregate with all planned denominators. Scheduler completion alone does not establish usable response coverage.", ""])
            atomic_bytes(markdown_path, (markdown_path.read_text() + "\n".join(lines)).encode())
        completion["status"] = "complete" if completion["plan_complete"] else "finalized_incomplete"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        completion["errors"].append(str(exc))
        completion["status"] = "finalization_error"
        completion["plan_complete"] = False
    completion["finished_at"] = utc_now()
    write_json(completion_path, completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    completion = finalize_run(args.run_dir)
    print(json.dumps({"status": completion["status"], "plan_complete": completion["plan_complete"],
                      "completion": str(args.run_dir.resolve() / "completion.json")}, indent=2))
    # Missing model outputs are a documented result, not a CPU finalizer crash.
    return 1 if completion["status"] == "finalization_error" or not completion.get("aggregate_succeeded") else 0


if __name__ == "__main__":
    raise SystemExit(main())
