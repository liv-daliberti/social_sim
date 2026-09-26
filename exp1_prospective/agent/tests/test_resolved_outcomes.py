from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from exp1_prospective.agent import evaluate_resolved_outcomes as resolved


ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = (
    ROOT / "exp1_prospective/data/results/resolved_market_snapshot_2026-08-24.json"
)
REPORT = (
    ROOT / "exp1_prospective/data/results/resolved_outcome_evaluation_2026-08-24.json"
)


def test_strict_terminal_rule() -> None:
    base = {"closed": True, "outcomes": '["Yes", "No"]'}
    assert resolved._terminal_outcome({**base, "outcomePrices": '["1", "0"]'}) == 1
    assert resolved._terminal_outcome({**base, "outcomePrices": '["0", "1"]'}) == 0
    assert resolved._terminal_outcome({**base, "outcomePrices": '["0.5", "0.5"]'}) is None
    assert resolved._terminal_outcome({**base, "closed": False, "outcomePrices": '["1", "0"]'}) is None


def test_hand_checkable_scores() -> None:
    result = resolved.score_probabilities(
        np.asarray([0.8, 0.4]),
        np.asarray([0.6, 0.7]),
        np.asarray([1.0, 0.0]),
        repetitions=1_000,
        seed=7,
    )
    assert result["accuracy"]["model_correct"] == 2
    assert result["accuracy"]["market_correct"] == 1
    assert result["disagreements"]["model_only_correct"] == 1
    assert result["disagreements"]["market_only_correct"] == 0
    assert result["brier"]["model"] == pytest.approx(0.1)
    assert result["brier"]["market"] == pytest.approx(0.325)
    assert result["brier"]["difference_model_minus_market"]["estimate"] == pytest.approx(-0.225)


def test_frozen_snapshot_and_report() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert snapshot["n_roster"] == snapshot["n_api_records"] == 100
    assert report["design"]["outcome_snapshot_sha256"] == resolved._sha256(SNAPSHOT)
    assert report["outcomes"] == {
        "n_roster": 100,
        "n_strict_terminal": 60,
        "n_yes": 21,
        "n_no": 39,
        "n_closed_nonterminal": 1,
        "closed_nonterminal_market_ids": ["540843"],
        "n_not_strict_terminal": 40,
    }
    assert set(report["per_model"]) == {
        "DeepSeek-V4-Pro",
        "claude-opus-4-8",
        "gpt-5.4",
        "llama3.1:8b",
        "llama3.1:70b",
        "llama3.3:70b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "qwen2.5:32b",
        "qwen2.5:72b",
    }


def test_rebuild_reproduces_frozen_point_estimates() -> None:
    rebuilt = resolved.build_report(
        resolved.DEFAULT_ROSTER,
        SNAPSHOT,
        resolved.EXPERIMENT_ROOT / "data/initial_forecasts",
        repetitions=500,
        seed=resolved.DEFAULT_SEED,
    )
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    for model, expected in frozen["per_model"].items():
        observed = rebuilt["per_model"][model]
        assert observed["n"] == expected["n"]
        assert observed["accuracy"]["model_correct"] == expected["accuracy"]["model_correct"]
        assert observed["brier"]["model"] == pytest.approx(expected["brier"]["model"])
        assert observed["log_loss"]["model"] == pytest.approx(expected["log_loss"]["model"])
