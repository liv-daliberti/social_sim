#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path


POLY = Path(__file__).resolve().parents[1]
REGISTRATION = POLY / "runs" / (
    "exp4_architecture_provider_sweep_opus5_registration_20260826T143622Z.json"
)
COMPLETION = POLY / "runs" / (
    "exp4_architecture_provider_sweep_opus5_complete_20260826T150235Z.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opus5_extension_was_registered_before_calls() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))

    assert registration["status"] == "registered"
    assert registration["timing_disclosure"]["parent_holdout_was_observed"] is True
    assert registration["timing_disclosure"]["opus5_holdout_calls_started"] is False
    assert registration["development_gate"]["calls"] == 2
    assert registration["added_model"]["deployment"] == "claude-opus-5"
    assert registration["added_model"]["temperature_supplied"] is False
    assert registration["added_model"]["max_output_tokens"] == 4_096
    assert registration["locked_evaluation"]["total_calls"] == 1_590
    assert registration["locked_evaluation"]["report_regardless_of_direction"]
    assert registration["frozen_data"]["holdout_n"] == 318

    for key in ("shared_runner", "deployment_adapter"):
        path = POLY.parents[1] / registration["software"][key]
        assert sha256(path) == registration["software"][f"{key}_sha256"]


def test_opus5_completion_binds_fail_closed_result() -> None:
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))

    assert completion["status"] == "complete"
    assert completion["development_gate"]["status"] == "pass"
    assert completion["execution"]["holdout_calls"] == 1_590
    assert completion["execution"]["transport_failures"] == 0
    assert completion["execution"]["successfully_parsed_draws"] == 1_578
    assert completion["execution"]["malformed_responses"] == 12
    assert completion["execution"]["complete_tasks"] == 309
    assert completion["execution"]["received_holdout_responses_retried"] == 0

    root = POLY.parents[1]
    registration = root / completion["registration"]["path"]
    summary = root / completion["result"]["summary"]
    raw = root / completion["result"]["raw"]
    assert sha256(registration) == completion["registration"]["sha256"]
    assert sha256(summary) == completion["result"]["summary_sha256"]
    assert sha256(raw) == completion["result"]["raw_sha256"]

    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["endpoint_summary"]["brier"] == 0.1557118028930818
    assert report["endpoint_summary"]["complete_task_parse_coverage"] == 309 / 318
    assert report["hosted_minus_market_brier"]["ci95_low"] > 0
