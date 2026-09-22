"""Completion preserves all prescribed arms and distinguishes invalid from absent."""
import json
from pathlib import Path

import pytest

from exp1_prospective.context_reversal import finalize_local_diagnostics as finalizer
from exp1_prospective.context_reversal.design import compile_plan
from exp1_prospective.context_reversal.direction_design import compile_plan as direction_plan
from exp1_prospective.context_reversal.materials import load_families


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def prepare_submission(tmp_path):
    parent_rows = compile_plan(load_families(), repeats=1)
    parent = tmp_path / "probability.jsonl"
    write_jsonl(parent, parent_rows)
    direction = tmp_path / "direction.jsonl"
    write_jsonl(direction, direction_plan(parent_rows, finalizer.sha256(parent)))
    jobs = [
        {"task": task, "model_key": key, "job_id": "103" if key.startswith("qwen3") else str(101 + index),
         "model_path": f"/local/models/{key}", "response_path": str(tmp_path / f"{key}.jsonl"),
         "results_dir": str(tmp_path / "results" / key)}
        for index, (task, key) in enumerate(finalizer.PRESCRIBED_ARMS)
    ]
    submission = {
        "run_id": "fixture_local_diagnostics", "jobs": jobs,
        "probability_design": str(parent), "probability_design_sha256": finalizer.sha256(parent),
        "direction_design": str(direction), "direction_design_sha256": finalizer.sha256(direction),
        "code_sha256": {name: finalizer.sha256(finalizer.SOURCE_DIR / name) for name in finalizer.REQUIRED_CODE},
    }
    finalizer.write_json(tmp_path / "submission.json", submission)
    return submission


def test_all_missing_runs_every_real_cpu_analysis_and_retains_1360_denominator(tmp_path):
    submission = prepare_submission(tmp_path)
    report = finalizer.finalize_run(tmp_path, scheduler=False)
    assert report["analysis_complete"], report
    assert report["status"] == "finalized_incomplete"
    assert not report["record_complete"] and not report["all_valid"]
    assert report["coverage_totals"] == {
        "expected": 1360, "arms_with_coverage": 5, "coverage_complete": True, "records_with_unknown_coverage": 0,
        "present": 0, "valid": 0, "missing_record": 1360, "invalid_present": 0,
    }
    assert len(report["arms"]) == 5
    for job in submission["jobs"]:
        assert not Path(job["response_path"]).exists()
        arm = report["arms"][f"{job['task']}:{job['model_key']}"]
        assert Path(arm["analysis_input"]).parent == tmp_path / "missing_responses"
        assert Path(arm["summary_path"]).is_file()
        assert arm["failure_counts"]["missing_record"] == finalizer.EXPECTED[job["task"]]
    markdown = (tmp_path / "summary.md").read_text()
    assert "1360" in markdown and "Sign accuracy" in markdown
    assert "Paired reversal" in markdown and "Broken-link diagnostic" in markdown
    for task, key in finalizer.PRESCRIBED_ARMS:
        assert f"| {task} | {key} |" in markdown


def test_all_records_present_with_generation_errors_reports_complete_with_errors(tmp_path):
    submission = prepare_submission(tmp_path)
    for job in submission["jobs"]:
        task = job["task"]
        units = [json.loads(line) for line in Path(submission[f"{task}_design"]).read_text().splitlines()]
        conditions = ("new_news", "no_news", "repeated_news")
        if task == "probability":
            conditions = ("baseline", *conditions)
        rows = []
        for unit in units:
            for condition in conditions:
                rows.append({
                    **{key: unit[key] for key in ("trial_id", "family_id", "domain", "context_id", "repeat", "material_status")},
                    "model_key": job["model_key"], "condition": condition, "status": "generation_error",
                    "stage": "direction" if task == "direction" else "baseline" if condition == "baseline" else "update",
                    "input_sha256": submission[f"{task}_design_sha256"], "probability": None, "direction": None, "raw": "",
                })
        write_jsonl(Path(job["response_path"]), rows)
    report = finalizer.finalize_run(tmp_path, scheduler=False)
    assert report["analysis_complete"], report
    assert report["status"] == "complete_with_errors"
    assert report["record_complete"] and not report["all_valid"]
    assert report["coverage_totals"]["present"] == 1360
    assert report["coverage_totals"]["valid"] == 0
    assert report["coverage_totals"]["invalid_present"] == 1360
    assert report["coverage_totals"]["missing_record"] == 0
    for arm in report["arms"].values():
        assert arm["analysis_input_unchanged"]
        assert arm["record_complete"] and not arm["all_valid"]
        assert arm["failure_counts"]["reason:generation_error"] == arm["expected_records"]


def test_complete_records_with_invalid_values_are_not_all_valid():
    counts = {"expected": 320, "present": 320, "valid": 319, "invalid_present": 1, "missing_record": 0}
    assert finalizer.coverage_flags(counts, 320) == {"record_complete": True, "all_valid": False}
    counts.update(valid=320, invalid_present=0)
    assert finalizer.coverage_flags(counts, 320) == {"record_complete": True, "all_valid": True}
    counts.update(present=319, missing_record=1, valid=319)
    assert finalizer.coverage_flags(counts, 320) == {"record_complete": False, "all_valid": False}


@pytest.mark.parametrize("corruption", ["code", "design", "missing_arm"])
def test_bad_frozen_provenance_or_roster_stops_before_commands(tmp_path, monkeypatch, corruption):
    submission = prepare_submission(tmp_path)
    if corruption == "code":
        submission["code_sha256"]["analyze.py"] = "0" * 64
    elif corruption == "design":
        submission["direction_design_sha256"] = "0" * 64
    else:
        submission["jobs"].pop()
    finalizer.write_json(tmp_path / "submission.json", submission)
    def forbidden(*args, **kwargs):
        pytest.fail("No scheduler or analyzer command may execute after failed provenance validation")
    monkeypatch.setattr(finalizer, "run_command", forbidden)
    report = finalizer.finalize_run(tmp_path)
    assert report["status"] == "finalization_error"
    assert not report["analysis_complete"]
    assert report["errors"]
    assert (tmp_path / "summary.md").exists()


def test_sacct_queries_shared_gpu_job_once(monkeypatch):
    commands = []
    def accounting(command, timeout):
        commands.append(command)
        return {"returncode": 0, "stdout": "101|COMPLETED|0:0|1:00|start|end|\n103|FAILED|1:0|1:00|start|end|\n", "stderr": ""}
    monkeypatch.setattr(finalizer, "run_command", accounting)
    result = finalizer.query_scheduler(["101", "103", "103", "103"])
    assert commands[0][commands[0].index("-j") + 1] == "101,103"
    assert set(result["jobs"]) == {"101", "103"}
    assert result["jobs"]["103"]["state"] == "FAILED"


def test_nonempty_missing_placeholder_is_never_overwritten(tmp_path):
    path = tmp_path / "missing_responses" / "arm.jsonl"
    path.parent.mkdir()
    path.write_text("preserve this existing record\n")
    with pytest.raises(ValueError, match="nonempty"):
        finalizer.missing_placeholder(tmp_path, "arm")
    assert path.read_text() == "preserve this existing record\n"


def test_corrupt_arm_is_retained_while_other_analyses_continue(tmp_path):
    submission = prepare_submission(tmp_path)
    corrupt = submission["jobs"][0]
    Path(corrupt["response_path"]).write_text("{invalid JSON\n")
    stale = Path(corrupt["results_dir"]) / "summary.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"stale": true}\n')
    report = finalizer.finalize_run(tmp_path, scheduler=False)
    assert not report["analysis_complete"]
    assert len(report["arms"]) == 5
    failed = report["arms"][f"direction:{corrupt['model_key']}"]
    assert not failed["analysis_succeeded"] and "analysis_error" in failed
    assert "metrics" not in failed and "coverage" not in failed
    assert report["coverage_totals"]["arms_with_coverage"] == 4
    assert sum(arm["analysis_succeeded"] for arm in report["arms"].values()) == 4
    assert Path(corrupt["response_path"]).read_text() == "{invalid JSON\n"
    assert "unknown / 240" in (tmp_path / "summary.md").read_text()
