#!/usr/bin/env python3
"""Audit six model-by-arm canaries before the full Coin City grid unlocks."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED = {(model, arm) for model in ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
            for arm in ("causal", "population_prior")}
REWARD_RE = re.compile(r"['\"]actor/rewards['\"]\s*:\s*([^,\s}\]]+)")


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


def parse_rate(path: Path) -> tuple[float, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    draws = sum(int(row["n_draws"]) for row in rows)
    parsed = sum(float(row["parse_rate"]) * int(row["n_draws"]) for row in rows)
    return parsed / draws, draws


def endpoint_contract(path: Path, *, temperature: float, draws: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != 1440:
        raise SystemExit(f"{path}: expected 1,440 endpoint records, got {len(records)}")
    for index, record in enumerate(records):
        outputs = record.get("output")
        if not isinstance(outputs, list) or len(outputs) != draws:
            raise SystemExit(f"{path}: record {index} does not contain exactly {draws} draws")
        if float(record.get("temperature", -1.0)) != temperature:
            raise SystemExit(f"{path}: record {index} has the wrong temperature")
        if record.get("structured_output") != "forecast_array":
            raise SystemExit(f"{path}: record {index} was not grammar constrained")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "protocol" / "canary_audit.json")
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    jobs = ledger.get("canaries", [])
    if {(row["model"], row["arm"]) for row in jobs} != EXPECTED or len(jobs) != 6:
        raise SystemExit("canary ledger is not the exact three-model by two-arm grid")
    job_ids = [str(row["job_id"]) for row in jobs]
    states = state_map(job_ids)
    incomplete = {job_id: states.get(job_id, "UNKNOWN") for job_id in job_ids
                  if states.get(job_id) != "COMPLETED"}
    if incomplete:
        raise SystemExit(f"canaries not complete: {incomplete}")

    details = []
    for row in jobs:
        job_id = str(row["job_id"])
        pattern = f"canary_{row['arm']}_{row['model']}_s42_*_j{job_id}"
        matches = sorted((ROOT / "reports").glob(pattern))
        if len(matches) != 1:
            raise SystemExit(f"job {job_id}: expected one report matching {pattern}, got {len(matches)}")
        report = matches[0]
        artifacts = {
            "greedy_raw": report / "greedy.json",
            "greedy_summary": report / "greedy.scores.summary.json",
            "stochastic_raw": report / "stochastic_n5.json",
            "stochastic_summary": report / "stochastic_n5.scores.summary.json",
        }
        for path in artifacts.values():
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"job {job_id}: missing {path}")
        endpoint_contract(artifacts["greedy_raw"], temperature=0.0, draws=1)
        endpoint_contract(artifacts["stochastic_raw"], temperature=0.7, draws=5)
        greedy_rate, greedy_draws = parse_rate(artifacts["greedy_summary"])
        stochastic_rate, stochastic_draws = parse_rate(artifacts["stochastic_summary"])
        if greedy_draws != 1440 or stochastic_draws != 7200:
            raise SystemExit(f"job {job_id}: incomplete endpoint draw counts")
        if min(greedy_rate, stochastic_rate) < 0.95:
            raise SystemExit(f"job {job_id}: parse gate failed {greedy_rate:.3f}/{stochastic_rate:.3f}")
        adapters = list(report.glob("debug_*/saved_models/step_*/adapter_model.safetensors"))
        if len(adapters) != 1 or not adapters[0].is_file() or adapters[0].stat().st_size == 0:
            raise SystemExit(f"job {job_id}: expected one nonempty saved adapter, got {adapters}")
        train_log = report / "train.log"
        try:
            rewards = [float(value) for value in
                       REWARD_RE.findall(train_log.read_text(errors="replace"))]
        except ValueError as error:
            raise SystemExit(f"job {job_id}: nonnumeric training reward: {error}") from error
        if len(rewards) < 2 or not np.isfinite(rewards).all() or np.ptp(rewards) < 1e-4:
            raise SystemExit(f"job {job_id}: training reward did not vary")
        details.append({
            "job_id": job_id, "model": row["model"], "arm": row["arm"],
            "report": str(report), "greedy_parse_rate": greedy_rate,
            "stochastic_parse_rate": stochastic_rate, "reward_range": float(np.ptp(rewards)),
            "adapter": str(adapters[0]), "greedy_endpoint": str(artifacts["greedy_raw"]),
        })
    output = {"protocol": "coin_city_structural_canary_gate_v2", "status": "passed",
              "ledger": str(args.ledger), "jobs": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Coin City canary gate passed: {len(details)} jobs")


if __name__ == "__main__":
    main()
