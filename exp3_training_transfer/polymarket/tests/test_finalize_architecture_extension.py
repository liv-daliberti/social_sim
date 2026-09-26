#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


POLY = Path(__file__).resolve().parents[1]
SCRIPTS = POLY / "scripts"
sys.path.insert(0, str(SCRIPTS))

import finalize_architecture_extension as finalizer  # noqa: E402


CURRENT_LLAMA_SUMMARY = (
    POLY / "reports" / "exp4_scale_llama3_1_8b_stochastic_j30889436.summary.json"
)
CURRENT_LLAMA_RAW = (
    POLY / "reports" / "exp4_scale_llama3_1_8b_stochastic_j30889436.jsonl"
)


def test_frozen_registration_and_all_seed_execution_chain_is_valid() -> None:
    registration, execution = finalizer.validate_registration(
        finalizer.REGISTRATION, finalizer.EXECUTION
    )

    assert set(registration["local_models"]) == {
        "qwen3_1_7b",
        "llama3_2_3b",
    }
    assert execution["scheduler"] == {
        "account": "mltheory",
        "partition": "all",
        "excluded_nodes": ["node206"],
        "scientific_settings_changed": False,
    }
    assert execution["qwen3_1_7b"]["evaluation"]["job_id"] == "30890110"
    assert execution["llama3_2_3b"]["evaluation"]["job_id"] == "30890115"


def test_task_universe_validator_accepts_existing_same_holdout_artifact() -> None:
    assert finalizer.validate_task_universe(CURRENT_LLAMA_RAW) == 318


def test_task_universe_validator_rejects_order_drift(tmp_path: Path) -> None:
    rows = finalizer.read_jsonl(CURRENT_LLAMA_RAW)
    rows[0], rows[1] = rows[1], rows[0]
    drifted = tmp_path / "drifted.jsonl"
    drifted.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="order or identity drifted"):
        finalizer.validate_task_universe(drifted)


def test_endpoint_validator_accepts_fail_closed_coverage_and_rejects_nan() -> None:
    summary = finalizer.load_json(CURRENT_LLAMA_SUMMARY)
    endpoint = summary["models"]["base"]

    finalizer.validate_endpoint("llama:base", endpoint)
    drifted = copy.deepcopy(endpoint)
    drifted["brier"] = float("nan")
    with pytest.raises(RuntimeError, match="invalid brier"):
        finalizer.validate_endpoint("llama:base", drifted)


def test_report_resolution_is_evaluation_job_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finalizer, "REPORTS", tmp_path)

    with pytest.raises(RuntimeError, match="missing completed qwen3_1_7b"):
        finalizer.resolve_report("qwen3_1_7b", "30890110")

    prefix = tmp_path / "exp4_scale_qwen3_1_7b_stochastic_j30890110"
    summary = prefix.with_suffix(".summary.json")
    raw = prefix.with_suffix(".jsonl")
    summary.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    assert finalizer.resolve_report("qwen3_1_7b", "30890110") == (
        summary,
        raw,
    )


def test_finalizer_checks_absolute_registered_snapshot_identity() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")

    assert 'summary.get("model") != registered["snapshot"]' in source
    assert 'summary.get("model") != registered["checkpoint"]' not in source
