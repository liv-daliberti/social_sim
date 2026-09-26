from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp1_prospective.agent import audit_orthogonal_abstention as abstention


ROOT = Path(__file__).resolve().parents[3]
FROZEN_REPORT = (
    ROOT / "exp1_prospective/data/results/consistency_report_2026-06-17.json"
)
ARTIFACT = ROOT / "exp1_prospective/data/results/orthogonal_abstention.json"
TABLE = ROOT / "paper/tables/exp1_abstention_decomposition.tex"


@pytest.fixture(scope="module")
def frozen() -> dict:
    if not FROZEN_REPORT.exists():
        pytest.skip("frozen consistency report is not available")
    return json.loads(FROZEN_REPORT.read_text())


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("abstention artifact has not been generated")
    return json.loads(ARTIFACT.read_text())


def test_decomposition_identity_is_exact(frozen):
    """Sensitivity must equal the magnitude factor times the abstention factor."""
    blocks = abstention.load_market_arrays(frozen)
    for model in abstention.MODEL_ORDER:
        stats = abstention.statistics(blocks[model])
        product = stats["sensitivity_given_move"] * stats["abstention_factor"]
        assert product == pytest.approx(stats["sensitivity"], rel=1e-12)


def test_reproduces_frozen_sensitivity_ratio(frozen):
    """The enumeration must recover each published unconditional ratio.

    Claude Opus 4.8 and GPT-5.4 carry a handful of runs the frozen aggregate
    counted in markets absent from the per-market join, so the tolerance matches
    the residual documented in audit_selectivity_robustness.py.
    """
    blocks = abstention.load_market_arrays(frozen)
    for model in abstention.MODEL_ORDER:
        published = frozen["per_model"][model]["anchoring"]["sensitivity_ratio"]
        reproduced = abstention.statistics(blocks[model])["sensitivity"]
        assert reproduced == pytest.approx(published, abs=0.3)


def test_directional_abstention_is_negligible(frozen):
    """The abstention factor is carried by the orthogonal arm alone."""
    blocks = abstention.load_market_arrays(frozen)
    for model in abstention.MODEL_ORDER:
        stats = abstention.statistics(blocks[model])
        assert stats["dir_abstention"] < 0.02
        assert stats["orth_abstention"] > stats["dir_abstention"]


def test_artifact_matches_recomputed_point_estimates(frozen, artifact):
    blocks = abstention.load_market_arrays(frozen)
    assert artifact["record_source"] == FROZEN_REPORT.name
    assert artifact["excluded_models"] == ["qwen2.5:7b"]
    for model in abstention.MODEL_ORDER:
        stored = artifact["per_model"][model]
        stats = abstention.statistics(blocks[model])
        for key in ("orth_abstention", "sensitivity", "sensitivity_given_move"):
            assert stored[key]["estimate"] == pytest.approx(stats[key], rel=1e-9)
            assert stored[key]["ci_lower"] <= stored[key]["estimate"]
            assert stored[key]["estimate"] <= stored[key]["ci_upper"]


def test_table_covers_the_full_scale_valid_roster(artifact):
    if not TABLE.exists():
        pytest.skip("paper table has not been generated")
    body = TABLE.read_text()
    for model in abstention.MODEL_ORDER:
        assert abstention.LABELS[model] in body
    assert "qwen2.5:7b" not in body
    assert body.count(r"\\") >= len(abstention.MODEL_ORDER)
