#!/usr/bin/env python3
"""Fail-closed engineering audit for the v2.1 large-model canaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
REPORTS = FAMILY / "reports"
PROTOCOL = "c3_mechanism_scale_v2_1"
MODELS = {"qwen3_14b", "qwen3_32b", "llama3_1_70b"}
DISCLOSURES = {"disclosed", "undisclosed"}
MIN_PARSE = 0.95
MAX_TRUNCATION = 0.05
TRAJECTORIES_PER_ROUND = 128


def slurm_states(job_ids: list[str]) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
            "--format=JobIDRaw,State",
        ],
        text=True,
    )
    states = {}
    for line in output.splitlines():
        if line.strip():
            job_id, state = line.split("|", 1)
            states[job_id] = state.split()[0].split("+")[0]
    return states


def overall_parse_rate(path: Path) -> float:
    rows = json.loads(path.read_text(encoding="utf-8"))
    total = sum(int(row["n_draws"]) for row in rows)
    parsed = sum(float(row["parse_rate"]) * int(row["n_draws"]) for row in rows)
    if total <= 0:
        raise AssertionError(f"empty score summary: {path}")
    return parsed / total


def require_complete(job_ids: list[str], states: dict[str, str] | None) -> dict[str, str]:
    observed = states if states is not None else slurm_states(job_ids)
    incomplete = {
        job_id: observed.get(job_id, "UNKNOWN")
        for job_id in job_ids
        if observed.get(job_id) != "COMPLETED"
    }
    if incomplete:
        raise AssertionError(f"large-model gate jobs incomplete: {incomplete}")
    return observed


def audit(
    ledger_path: Path,
    *,
    reports: Path = REPORTS,
    states: dict[str, str] | None = None,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("protocol") != PROTOCOL or ledger.get("stage") != "canary":
        raise AssertionError("not a v2.1 canary ledger")
    training = [row for row in ledger["submissions"] if row["kind"] == "canary"]
    bases = [row for row in ledger["submissions"] if row["kind"] == "base_evaluation"]
    expected = {(model, disclosure) for model in MODELS for disclosure in DISCLOSURES}
    if len(training) != 6 or {(r["model_key"], r["disclosure"]) for r in training} != expected:
        raise AssertionError("canary roster is not the exact six-cell grid")
    if len(bases) != 6 or {(r["model_key"], r["disclosure"]) for r in bases} != expected:
        raise AssertionError("base roster is not the exact six-cell grid")
    all_jobs = [str(row["job_id"]) for row in training + bases]
    observed = require_complete(all_jobs, states)

    details = []
    for row in training:
        job_id = str(row["job_id"])
        roots = sorted(reports.glob(f"scale_v2_1_canary_*_j{job_id}"))
        if len(roots) != 1:
            raise AssertionError(f"job {job_id}: expected one canary report, got {roots}")
        root = roots[0]
        adapters = sorted(root.glob("debug_*/saved_models/step_*"))
        if not adapters:
            raise AssertionError(f"job {job_id}: adapter missing")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapters[-1] / name).is_file():
                raise AssertionError(f"job {job_id}: {name} missing")
        greedy = sorted(root.glob("debug_*/eval_results/*.scores.summary.json"))
        stochastic = sorted(root.glob("stochastic_n1.scores.summary.json"))
        if not greedy or not stochastic:
            raise AssertionError(f"job {job_id}: endpoint score summaries missing")
        greedy_rate = overall_parse_rate(greedy[-1])
        stochastic_rate = overall_parse_rate(stochastic[-1])
        if min(greedy_rate, stochastic_rate) < MIN_PARSE:
            raise AssertionError(f"job {job_id}: syntax parse gate failed")
        log = (root / "train.log").read_text(errors="replace")
        rewards = [float(value) for value in re.findall(r"actor reward ([0-9.]+)", log)]
        if len(rewards) < 2 or not np.all(np.isfinite(rewards)) or np.std(rewards) <= 1e-4:
            raise AssertionError(f"job {job_id}: reward trace missing or collapsed")
        no_eos = [
            float(value)
            for value in re.findall(r"'actor/no_eos_count': ([0-9.]+)", log)
        ]
        truncation = max(no_eos) / TRAJECTORIES_PER_ROUND if no_eos else 1.0
        if truncation > MAX_TRUNCATION:
            raise AssertionError(f"job {job_id}: truncation rate {truncation:.3f}")
        details.append(
            {
                "kind": "canary",
                "job_id": job_id,
                "model_key": row["model_key"],
                "disclosure": row["disclosure"],
                "report": str(root),
                "adapter": str(adapters[-1]),
                "greedy_parse_rate": greedy_rate,
                "stochastic_parse_rate": stochastic_rate,
                "reward_trace_std": float(np.std(rewards)),
                "maximum_training_truncation_rate": truncation,
            }
        )

    for row in bases:
        job_id = str(row["job_id"])
        root = reports / f"scale_v2_1_base_{row['disclosure']}_{row['model_key']}_j{job_id}"
        summaries = [
            root / "greedy.scores.summary.json",
            root / "stochastic_n5.scores.summary.json",
        ]
        if any(not path.is_file() for path in summaries):
            raise AssertionError(f"job {job_id}: base summaries missing")
        rates = [overall_parse_rate(path) for path in summaries]
        if min(rates) < MIN_PARSE:
            raise AssertionError(f"job {job_id}: base syntax parse gate failed")
        details.append(
            {
                "kind": "base_evaluation",
                "job_id": job_id,
                "model_key": row["model_key"],
                "disclosure": row["disclosure"],
                "report": str(root),
                "greedy_parse_rate": rates[0],
                "stochastic_parse_rate": rates[1],
            }
        )

    return {
        "status": "pass",
        "protocol": PROTOCOL,
        "ledger": str(ledger_path),
        "states": observed,
        "jobs": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    result = audit(args.ledger)
    output = args.write or HERE / "runs" / "canary_audit_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
