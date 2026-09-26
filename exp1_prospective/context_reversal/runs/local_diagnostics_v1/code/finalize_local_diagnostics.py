#!/usr/bin/env python3
"""CPU-only finalization of all five prescribed local diagnostic arms.

Run after the GPU jobs terminate. Missing response files are represented only in
run-local missing_responses files; planned records and failed arms are retained.
Analysis executes the live modules only after checking frozen code hashes.
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

SOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY = SOURCE_DIR.parents[1]
DIRECTION_MODELS = ("qwen2_5_72b_instruct", "llama3_1_70b_instruct", "qwen3_32b")
PROBABILITY_MODELS = ("qwen3_32b_unconstrained_nonthinking", "qwen3_32b_thinking")
PRESCRIBED_ARMS = tuple(("direction", key) for key in DIRECTION_MODELS) + tuple(
    ("probability", key) for key in PROBABILITY_MODELS
)
EXPECTED = {"direction": 240, "probability": 320}
EXPECTED_TOTAL = 1360
REQUIRED_CODE = {"analyze.py", "analyze_direction.py", "direction_design.py", "run_local.py",
                 "run_direction_local.py", "design.py", "materials.py", "finalize_local_diagnostics.py"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_command(command: list[str], timeout: int = 600) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=REPOSITORY, text=True, capture_output=True,
                                timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def query_scheduler(job_ids: list[str]) -> dict[str, Any]:
    """Accounting is descriptive; unavailable sacct cannot invalidate coverage."""
    unique = list(dict.fromkeys(job_ids))
    query = run_command(["sacct", "-n", "-P", "-X", "-j", ",".join(unique),
                         "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End"], timeout=30)
    jobs = {key: {"state": "UNKNOWN", "accounting_row_present": False} for key in unique}
    if query["returncode"] == 0:
        for line in query["stdout"].splitlines():
            fields = line.rstrip("|").split("|")
            if len(fields) >= 6 and fields[0] in jobs:
                jobs[fields[0]] = {
                    "state": fields[1].strip().split(" ", 1)[0].rstrip("+"),
                    "raw_state": fields[1], "exit_code": fields[2], "elapsed": fields[3],
                    "started_at": fields[4], "ended_at": fields[5], "accounting_row_present": True,
                }
    return {"queried_at": utc_now(), "query": query, "jobs": jobs}


def validate_submission(submission: dict) -> tuple[dict, dict]:
    jobs = submission["jobs"]
    indexed = {(job["task"], job["model_key"]): job for job in jobs}
    if len(jobs) != 5 or set(indexed) != set(PRESCRIBED_ARMS):
        raise ValueError("Submission must contain exactly all five prescribed task/model arms")
    for field in ("response_path", "results_dir"):
        paths = [Path(job[field]) for job in jobs]
        if any(not path.is_absolute() for path in paths) or len(set(paths)) != 5:
            raise ValueError(f"Each arm requires a distinct absolute {field}")
    if any(not str(job["job_id"]).strip() or not job["model_path"] for job in jobs):
        raise ValueError("Each prescribed arm requires job_id and model_path")
    hashes = submission["code_sha256"]
    if not REQUIRED_CODE.issubset(hashes):
        raise ValueError("Frozen code hashes must include all analyzer dependencies: " + ", ".join(sorted(REQUIRED_CODE)))
    checked = {}
    for filename, expected in hashes.items():
        if Path(filename).name != filename:
            raise ValueError("code_sha256 keys must be source filenames")
        actual = sha256(SOURCE_DIR / filename)
        if actual != expected:
            raise ValueError(f"Frozen analysis code SHA-256 mismatch: {filename}")
        checked[filename] = actual
    designs = {}
    for task in EXPECTED:
        path = Path(submission[f"{task}_design"])
        if not path.is_absolute():
            raise ValueError(f"{task}_design must be absolute")
        actual = sha256(path)
        if actual != submission[f"{task}_design_sha256"]:
            raise ValueError(f"Frozen {task} design SHA-256 mismatch")
        designs[task] = {"path": str(path), "sha256": actual}
    return indexed, {"code_sha256": checked, "designs": designs}


def coverage_flags(coverage: dict, expected: int) -> dict[str, bool]:
    complete = (coverage.get("expected") == expected and coverage.get("present") == expected
                and coverage.get("missing_record", 0) == 0)
    return {"record_complete": complete,
            "all_valid": complete and coverage.get("valid") == expected and coverage.get("invalid_present", 0) == 0}


def missing_placeholder(run_dir: Path, name: str) -> Path:
    path = run_dir / "missing_responses" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes():
        raise ValueError(f"Refusing to overwrite a nonempty missing-response placeholder: {path}")
    if not path.exists():
        path.write_bytes(b"")
    return path


def failure_counts(coverage: dict) -> dict:
    return {key: value for key, value in coverage.items()
            if key not in {"expected", "planned", "present", "valid", "ok", "reason:ok"}}


def metric_text(metric: dict | None) -> str:
    if not metric or metric.get("estimate") is None:
        return "unavailable"
    text = f"{100 * metric['estimate']:.1f}% ({metric['n_success']}/{metric['n_planned']})"
    ci = metric.get("ci95")
    if ci is not None:
        text += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return text


def markdown_summary(report: dict) -> str:
    lines = ["# Local context-reversal diagnostics", "",
             "Exploratory diagnostics on unchanged development materials. The existing human materials review remains complete; these results do not claim new review coverage.", "",
             "All five prescribed arms are retained. Missing and invalid responses fail planned-denominator binary metrics; no forecast is imputed. Intervals are descriptive family-bootstrap intervals. Results do not select a model.", "",
             f"Status: **{report['status']}**. Record complete: **{str(report['record_complete']).lower()}**. All valid: **{str(report['all_valid']).lower()}**. Expected responses: **{EXPECTED_TOTAL}**.", "",
             "Probability sign and paired-reversal columns use the original raw new-news-minus-baseline outcomes. Direction columns use the categorical companion task; their scores measure different responses.", "",
             "| Task | Model / arm | Present / planned | Valid | Missing | Invalid present | Sign accuracy | Paired reversal | Broken-link diagnostic |",
             "|---|---|---:|---:|---:|---:|---|---|---|"]
    for task, key in PRESCRIBED_ARMS:
        arm = report["arms"].get(f"{task}:{key}", {})
        coverage, summary = arm.get("coverage", {}), arm.get("metrics", {})
        source = summary.get("primary" if task == "probability" else "companion", {})
        sign = source.get("direction_correct", {}).get("pooled")
        pair = source.get("paired_reversal", {}).get("both_directions_correct")
        if task == "probability":
            stability = summary.get("broken_stability", {})
            broken = "; ".join(f"{name}: {value['status']} ({value.get('reason', 'unspecified')})"
                               for name, value in stability.items()) or "unavailable"
        else:
            broken = "unchanged: " + metric_text(summary.get("by_context_condition", {}).get("broken", {}).get("new_news", {}).get("accuracy"))
        lines.append(f"| {task} | {key} | {coverage.get('present', 'unknown')} / {EXPECTED[task]} | {coverage.get('valid', 'unknown')} | {coverage.get('missing_record', 'unknown')} | {coverage.get('invalid_present', 'unknown')} | {metric_text(sign)} | {metric_text(pair)} | {broken} |")
    lines.extend(["", "Probability broken-link stability requires complete measurements, the signed-mean 90% interval strictly inside ±2 percentage points, and the upper 95% mean-absolute interval below 2 points. Nonsignificance is not equivalence.", "",
                  "## Response failures and analysis status", "",
                  "Invalid-present counts are totals; individual failure statuses below are their components, not additional records. An analysis error leaves coverage unknown instead of treating corrupt records as absent.", "",
                  "| Task | Model / arm | Analysis succeeded | Failure counts | Analysis error |", "|---|---|---|---|---|"])
    for task, key in PRESCRIBED_ARMS:
        arm = report["arms"].get(f"{task}:{key}", {})
        counts = "; ".join(f"{name}={value}" for name, value in sorted(arm.get("failure_counts", {}).items())) or "unavailable"
        error = str(arm.get("analysis_error", "none")).replace("|", "/").replace("\n", " ")
        lines.append(f"| {task} | {key} | {str(arm.get('analysis_succeeded', False)).lower()} | {counts} | {error} |")
    lines.extend(["", "## Direction control judgments", "",
                  "Masked new messages have no prespecified direction target. Their label counts and unclear rates remain in the per-arm analysis, and all control conditions below retain planned denominators.", "",
                  "| Model | Context | Condition | Accuracy | Unclear rate |", "|---|---|---|---|---|"])
    for key in DIRECTION_MODELS:
        summary = report["arms"].get(f"direction:{key}", {}).get("metrics", {})
        for context, conditions in summary.get("by_context_condition", {}).items():
            for condition, values in conditions.items():
                if condition == "new_news":
                    continue
                lines.append(f"| {key} | {context} | {condition} | {metric_text(values.get('accuracy'))} | {metric_text(values.get('unclear_rate'))} |")
    for error in report.get("errors", []):
        lines.extend(["", "Finalization error: " + str(error)])
    lines.extend(["", "Scheduler accounting is recorded separately from response completeness. A completed scheduler job alone does not establish valid responses. Per-arm summary paths, response hashes, analysis commands, and full metrics are retained in completion.json.", ""])
    return "\n".join(lines)


def finalize_run(run_dir: Path, *, scheduler: bool = True) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    completion_path = run_dir / "completion.json"
    report: dict[str, Any] = {
        "schema_version": "context_reversal_local_diagnostics_completion_v1",
        "run_dir": str(run_dir), "started_at": utc_now(), "status": "finalizing",
        "record_complete": False, "all_valid": False, "analysis_complete": False,
        "expected_records": EXPECTED_TOTAL, "frontier_api_calls": 0,
        "new_human_review_requested": False, "arms": {}, "errors": [],
    }
    write_json(completion_path, report)
    try:
        submission_path = run_dir / "submission.json"
        submission = json.loads(submission_path.read_text())
        report["run_id"] = submission["run_id"]
        report["submission_sha256"] = sha256(submission_path)
        indexed, provenance = validate_submission(submission)
        report.update(provenance)
        report["scheduler"] = query_scheduler([str(job["job_id"]) for job in indexed.values()]) if scheduler else {"skipped": True, "jobs": {}}
        for task, key in PRESCRIBED_ARMS:
            job = indexed[task, key]
            name = f"{task}:{key}"
            response = Path(job["response_path"])
            arm = {"task": task, "model_key": key, "model_path": job["model_path"],
                   "job_id": str(job["job_id"]), "raw_response_path": str(response),
                   "raw_response_present": response.is_file(), "expected_records": EXPECTED[task],
                   "analysis_succeeded": False, "record_complete": False, "all_valid": False}
            report["arms"][name] = arm
            try:
                analysis_input = response if response.is_file() else missing_placeholder(run_dir, name.replace(":", "_"))
                arm["analysis_input"] = str(analysis_input)
                arm["analysis_input_sha256"] = sha256(analysis_input)
                if not response.is_file():
                    arm["missing_input_representation"] = "empty_placeholder_preserving_all_planned_denominators"
                module = "analyze_direction" if task == "direction" else "analyze"
                command = [sys.executable, "-m", f"exp1_prospective.context_reversal.{module}",
                           "--design", report["designs"][task]["path"], "--responses", str(analysis_input),
                           "--model-key", key, "--output", job["results_dir"]]
                reference = job.get("probability_reference_path")
                if task == "direction" and reference:
                    reference_path = Path(reference)
                    reference_input = reference_path if reference_path.is_file() else missing_placeholder(run_dir, f"probability_reference_{key}")
                    arm["probability_reference"] = {"path": str(reference_path), "present": reference_path.is_file(),
                                                    "analysis_input": str(reference_input), "sha256": sha256(reference_input)}
                    command += ["--probability-design", report["designs"]["probability"]["path"],
                                "--probability-responses", str(reference_input)]
                    if job.get("probability_reference_design_path"):
                        command += ["--probability-source-design", job["probability_reference_design_path"]]
                print(f"Analyzing {name}; response present={arm['raw_response_present']}", flush=True)
                # Recheck before every module execution, not just before accounting.
                validate_submission(submission)
                result = run_command(command)
                arm["analysis_process"] = result
                arm["analysis_input_unchanged"] = sha256(analysis_input) == arm["analysis_input_sha256"]
                if result["returncode"] != 0:
                    raise ValueError(f"Analyzer exited {result['returncode']}; see analysis_process")
                if not arm["analysis_input_unchanged"]:
                    raise ValueError("Response input changed during analysis")
                if "probability_reference" in arm:
                    reference_info = arm["probability_reference"]
                    reference_info["unchanged"] = sha256(Path(reference_info["analysis_input"])) == reference_info["sha256"]
                    if not reference_info["unchanged"]:
                        raise ValueError("Probability reference changed during analysis")
                summary_path = Path(job["results_dir"]) / "summary.json"
                summary = json.loads(summary_path.read_text())
                if summary.get("model_key") != key or summary.get("provenance", {}).get("design", {}).get("sha256") != report["designs"][task]["sha256"]:
                    raise ValueError("Analysis model/design provenance mismatch")
                coverage = summary["coverage"]["totals"]
                if coverage.get("expected") != EXPECTED[task]:
                    raise ValueError("Analysis changed the prescribed response denominator")
                arm.update(coverage_flags(coverage, EXPECTED[task]))
                arm.update({"analysis_succeeded": True, "summary_path": str(summary_path),
                            "summary_sha256": sha256(summary_path), "coverage": coverage,
                            "failure_counts": failure_counts(coverage), "metrics": summary})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                arm["analysis_error"] = str(exc)
            write_json(completion_path, report)
        arms = list(report["arms"].values())
        report["analysis_complete"] = all(arm["analysis_succeeded"] for arm in arms)
        report["record_complete"] = all(arm["record_complete"] for arm in arms)
        report["all_valid"] = all(arm["all_valid"] for arm in arms)
        report["coverage_totals"] = {
            "expected": EXPECTED_TOTAL, "arms_with_coverage": sum("coverage" in arm for arm in arms),
            "coverage_complete": report["analysis_complete"],
            "records_with_unknown_coverage": sum(arm["expected_records"] for arm in arms if "coverage" not in arm),
            **{field: sum(arm.get("coverage", {}).get(field, 0) for arm in arms)
               for field in ("present", "valid", "missing_record", "invalid_present")},
        }
        report["status"] = ("finalized_complete" if report["all_valid"] else
                            "complete_with_errors" if report["record_complete"] else "finalized_incomplete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
        report["status"] = "finalization_error"
    report["finished_at"] = utc_now()
    write_json(completion_path, report)
    atomic_write(run_dir / "summary.md", markdown_summary(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--no-scheduler", action="store_true", help="Skip optional sacct accounting")
    args = parser.parse_args(argv)
    report = finalize_run(args.run_dir, scheduler=not args.no_scheduler)
    print(json.dumps({key: report[key] for key in ("status", "record_complete", "all_valid", "analysis_complete")}, indent=2))
    return 0 if report["analysis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
