#!/usr/bin/env python3
"""Gate full C3/Polymarket launches on registered training and base canaries."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
DEFAULT_OUTPUT = ROOT / "protocol" / "canary_audit.json"
MIN_OVERALL_PARSE_RATE = 0.95
MAX_TRAIN_TRUNCATION_RATE = 0.05
TRAJECTORIES_PER_ROUND = 128
EXPECTED_BASE_BLOCKS = {
    "mixed_composition",
    "nonlinear_composition",
    "parameter_extrapolation",
    "topology_composition",
}
EXPECTED_K = {3, 6, 9}
EXPECTED_BASE_TASKS_PER_CELL = 60


def latest_canary_ledger() -> Path:
    candidates = sorted(RUNS.glob("c3_mechanism_*.json"), reverse=True)
    for path in candidates:
        ledger = json.loads(path.read_text())
        training = [item for item in ledger["submissions"] if item["kind"] == "training"]
        if len(training) == 6 and all(item["arm"] == "causal_family" for item in training):
            return path
    raise AssertionError("no six-training-job canary ledger found")


def is_base_canary_ledger(ledger: dict) -> bool:
    training = [
        item for item in ledger["submissions"]
        if item["kind"] == "training"
    ]
    bases = [
        item for item in ledger["submissions"]
        if item["kind"] == "base_evaluation"
    ]
    return (
        len(bases) == 6
        and len(training) == 6
        and all(item.get("arm") == "causal_family" and item.get("seed") == 42
                for item in training)
    )


def latest_base_ledger() -> Path:
    candidates = sorted(RUNS.glob("c3_mechanism_*.json"), reverse=True)
    for path in candidates:
        if is_base_canary_ledger(json.loads(path.read_text())):
            return path
    raise AssertionError("no six-base-job canary ledger found")


def slurm_states(job_ids: list[str]) -> dict[str, str]:
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
               "--format=JobIDRaw,State"]
    output = subprocess.check_output(command, text=True)
    states = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        states[job_id] = state.split()[0].split("+")[0]
    return states


def overall_parse_rate(path: Path) -> float:
    summaries = json.loads(path.read_text())
    total = sum(int(item["n_draws"]) for item in summaries)
    parsed = sum(float(item["parse_rate"]) * int(item["n_draws"])
                 for item in summaries)
    return parsed / total


def _base_summary_contract(
    path: Path,
    *,
    model: str,
    disclosure: str,
    decode: str,
    draws_per_task: int,
) -> tuple[float, int]:
    summaries = json.loads(path.read_text())
    expected_cells = {(block, k) for block in EXPECTED_BASE_BLOCKS for k in EXPECTED_K}
    actual_cells = {(item.get("block"), item.get("k")) for item in summaries}
    if len(summaries) != len(expected_cells) or actual_cells != expected_cells:
        raise AssertionError(f"{path}: incomplete structural-OOD block/k grid")
    expected_draws = EXPECTED_BASE_TASKS_PER_CELL * draws_per_task
    for item in summaries:
        identity = (
            item.get("model"), item.get("disclosure"), item.get("arm"),
            item.get("seed"), item.get("decode"), item.get("n_tasks"),
            item.get("n_draws"),
        )
        expected_identity = (
            model, disclosure, "base", 0, decode,
            EXPECTED_BASE_TASKS_PER_CELL, expected_draws,
        )
        if identity != expected_identity:
            raise AssertionError(
                f"{path}: unexpected base-summary identity/counts {identity!r}"
            )
    total_draws = sum(int(item["n_draws"]) for item in summaries)
    expected_total = len(expected_cells) * expected_draws
    if total_draws != expected_total:
        raise AssertionError(
            f"{path}: expected {expected_total} scored draws, found {total_draws}"
        )
    return overall_parse_rate(path), total_draws


def audit_base_evaluations(
    ledger_path: Path,
    *,
    reports: Path = REPORTS,
    states: dict[str, str] | None = None,
) -> dict:
    """Audit all six untrained-base cells, resolving recorded scheduler replacements."""
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("protocol") != "c3_mechanism_disclosure_v4":
        raise AssertionError("base ledger does not use the structured-decoding v4 protocol")
    decoding = ledger.get("structured_decoding", {})
    if (decoding.get("kind"), decoding.get("backend"), decoding.get("value_constraints")) != (
            "gbnf", "xgrammar", "none"):
        raise AssertionError("base ledger does not register syntax-only GBNF decoding")

    jobs = [item for item in ledger["submissions"] if item["kind"] == "base_evaluation"]
    expected_cells = {(disclosure, model) for disclosure in ("disclosed", "undisclosed")
                      for model in ("qwen3_4b", "qwen3_8b", "llama3_1_8b")}
    actual_cells = {(item["disclosure"], item["model"]) for item in jobs}
    if len(jobs) != 6 or actual_cells != expected_cells:
        raise AssertionError("ledger is not the registered six-cell base-evaluation grid")

    job_id_map = {
        str(old): str(new)
        for old, new in ledger.get("base_evaluation_replacement", {})
        .get("job_id_map", {}).items()
    }
    original_ids = {str(item["job_id"]) for item in jobs}
    if set(job_id_map) - original_ids:
        raise AssertionError("base replacement map contains an unregistered source job")
    if any(not (old.isdigit() and new.isdigit()) for old, new in job_id_map.items()):
        raise AssertionError("base replacement job IDs must be numeric")
    if len(set(job_id_map.values())) != len(job_id_map):
        raise AssertionError("base replacement target job IDs must be unique")

    resolved_ids = [job_id_map.get(str(item["job_id"]), str(item["job_id"])) for item in jobs]
    observed_states = states if states is not None else slurm_states(resolved_ids)
    incomplete = {
        job_id: observed_states.get(job_id, "UNKNOWN")
        for job_id in resolved_ids
        if observed_states.get(job_id) != "COMPLETED"
    }
    if incomplete:
        raise AssertionError(f"base canaries have not all completed successfully: {incomplete}")

    details = []
    for item, job_id in zip(jobs, resolved_ids):
        environment = item.get("environment", {})
        expected_environment = {
            "MAX_MODEL_LEN": "3072",
            "MAX_TOKENS": "192",
            "STOCHASTIC_N": "5",
            "STOCHASTIC_TEMP": "0.7",
            "STRUCTURED_OUTPUT": "forecast_array",
        }
        mismatches = {
            key: environment.get(key)
            for key, value in expected_environment.items()
            if environment.get(key) != value
        }
        if mismatches:
            raise AssertionError(f"job {job_id}: unexpected base configuration {mismatches}")

        root = reports / f"base_{item['disclosure']}_{item['model']}_j{job_id}"
        if not root.is_dir():
            raise AssertionError(f"job {job_id}: base report directory missing: {root}")
        greedy_raw = root / "greedy.json"
        stochastic_raw = root / "stochastic_n5.json"
        greedy = root / "greedy.scores.summary.json"
        stochastic = root / "stochastic_n5.scores.summary.json"
        for artifact in (greedy_raw, stochastic_raw, greedy, stochastic):
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise AssertionError(f"job {job_id}: missing or empty artifact {artifact}")
        greedy_rate, greedy_draws = _base_summary_contract(
            greedy,
            model=item["model"],
            disclosure=item["disclosure"],
            decode="greedy",
            draws_per_task=1,
        )
        stochastic_rate, stochastic_draws = _base_summary_contract(
            stochastic,
            model=item["model"],
            disclosure=item["disclosure"],
            decode="stochastic",
            draws_per_task=5,
        )
        if min(greedy_rate, stochastic_rate) < MIN_OVERALL_PARSE_RATE:
            raise AssertionError(
                f"job {job_id}: base parse gate failed "
                f"({greedy_rate:.3f}, {stochastic_rate:.3f})"
            )
        details.append({
            "job_id": job_id,
            "replaces_job_id": (
                str(item["job_id"]) if job_id != str(item["job_id"]) else None
            ),
            "disclosure": item["disclosure"],
            "model": item["model"],
            "report": str(root),
            "greedy_scored_draws": greedy_draws,
            "stochastic_scored_draws": stochastic_draws,
            "greedy_overall_parse_rate": greedy_rate,
            "stochastic_overall_parse_rate": stochastic_rate,
        })
    return {
        "status": "pass",
        "protocol": "c3_mechanism_base_canary_gate_v4",
        "ledger": str(ledger_path),
        "job_id_map": job_id_map,
        "jobs": details,
    }


def audit(ledger_path: Path) -> dict:
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("protocol") != "c3_mechanism_disclosure_v4":
        raise AssertionError("canary ledger does not use the structured-decoding v4 protocol")
    decoding = ledger.get("structured_decoding", {})
    if (decoding.get("kind"), decoding.get("backend"), decoding.get("value_constraints")) != (
            "gbnf", "xgrammar", "none"):
        raise AssertionError("canary ledger does not register syntax-only GBNF decoding")
    jobs = [item for item in ledger["submissions"] if item["kind"] == "training"]
    expected_cells = {(disclosure, model) for disclosure in ("disclosed", "undisclosed")
                      for model in ("qwen3_4b", "qwen3_8b", "llama3_1_8b")}
    actual_cells = {(item["disclosure"], item["model"]) for item in jobs}
    if len(jobs) != 6 or actual_cells != expected_cells:
        raise AssertionError("ledger is not the registered six-cell synthetic canary grid")
    job_ids = [str(item["job_id"]) for item in jobs]
    states = slurm_states(job_ids)
    incomplete = {job_id: states.get(job_id, "UNKNOWN") for job_id in job_ids
                  if states.get(job_id) != "COMPLETED"}
    if incomplete:
        raise AssertionError(f"canaries have not all completed successfully: {incomplete}")

    details = []
    for item in jobs:
        job_id = str(item["job_id"])
        roots = sorted(REPORTS.glob(f"*_j{job_id}"))
        if len(roots) != 1:
            raise AssertionError(f"job {job_id}: expected one report directory, found {roots}")
        root = roots[0]
        adapters = sorted(root.glob("debug_*/saved_models/step_*"))
        if not adapters or not (adapters[-1] / "adapter_config.json").is_file():
            raise AssertionError(f"job {job_id}: final adapter missing")
        greedy = sorted(root.glob("debug_*/eval_results/*.scores.summary.json"))
        stochastic = sorted(root.glob("stochastic_n*.scores.summary.json"))
        if not greedy or not stochastic:
            raise AssertionError(f"job {job_id}: greedy/stochastic score summaries missing")
        greedy_rate = overall_parse_rate(greedy[-1])
        stochastic_rate = overall_parse_rate(stochastic[-1])
        if min(greedy_rate, stochastic_rate) < MIN_OVERALL_PARSE_RATE:
            raise AssertionError(
                f"job {job_id}: parse gate failed ({greedy_rate:.3f}, {stochastic_rate:.3f})"
            )
        log = (root / "train.log").read_text(errors="replace")
        rewards = [float(value) for value in re.findall(r"actor reward ([0-9.]+)", log)]
        if len(rewards) < 2 or float(np.std(rewards)) <= 1e-4:
            raise AssertionError(f"job {job_id}: reward trace is absent or collapsed")
        no_eos = [float(value) for value in re.findall(
            r"'actor/no_eos_count': ([0-9.]+)", log
        )]
        max_truncation_rate = max(no_eos) / TRAJECTORIES_PER_ROUND if no_eos else 1.0
        if max_truncation_rate > MAX_TRAIN_TRUNCATION_RATE:
            raise AssertionError(
                f"job {job_id}: training truncation rate {max_truncation_rate:.3f} exceeds "
                f"{MAX_TRAIN_TRUNCATION_RATE:.3f}"
            )
        details.append({
            "job_id": job_id, "disclosure": item["disclosure"], "model": item["model"],
            "adapter": str(adapters[-1].relative_to(ROOT.parent.parent)),
            "greedy_overall_parse_rate": greedy_rate,
            "stochastic_overall_parse_rate": stochastic_rate,
            "reward_trace_points": len(rewards), "reward_trace_std": float(np.std(rewards)),
            "maximum_no_eos_count": max(no_eos),
            "maximum_training_truncation_rate": max_truncation_rate,
        })
    return {"status": "pass", "protocol": "c3_mechanism_canary_gate_v4",
            "ledger": str(ledger_path.relative_to(ROOT.parent.parent)), "jobs": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--base-ledger", type=Path)
    parser.add_argument("--write", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = args.ledger or latest_canary_ledger()
    base_ledger = args.base_ledger or latest_base_ledger()
    result = audit(ledger)
    result["base_evaluations"] = audit_base_evaluations(base_ledger)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS -> {args.write}")


if __name__ == "__main__":
    main()
