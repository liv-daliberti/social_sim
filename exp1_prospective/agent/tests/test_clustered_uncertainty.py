from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp1_prospective.agent import evaluate_clustered_uncertainty as clustered


ROOT = Path(__file__).resolve().parents[3]
FROZEN_REPORT = (
    ROOT / "exp1_prospective/data/results/consistency_report_2026-06-17.json"
)
FROZEN_UNCERTAINTY = (
    ROOT / "exp1_prospective/data/results/clustered_movement_uncertainty.json"
)


def _toy_model_report() -> dict:
    markets = []
    for task_id, values in (
        ("market_b", (0.3, -0.4, -0.03)),
        ("market_a", (0.1, -0.2, 0.01)),
    ):
        counterfactuals = []
        for direction, delta in zip(clustered.DIRECTIONS, values):
            row = {
                "cf_id": f"{task_id}_{direction}_0",
                "direction": direction,
                "runs": [{"delta_yes_prob": delta}],
            }
            counterfactuals.append(row)
        markets.append(
            {
                "task_id": task_id,
                "n_updates": 3,
                "cf_results": counterfactuals,
            }
        )

    # Exercise the legacy-record recovery that the frozen report requires.
    del markets[0]["cf_results"][1]["direction"]
    return {
        "per_market": markets,
        "anchoring": {
            "pro_H1": {"mean_abs_delta": 0.2, "n": 2},
            "anti_H1": {"mean_abs_delta": 0.3, "n": 2},
            "orthogonal": {"mean_abs_delta": 0.02, "n": 2},
            "sensitivity_ratio": 12.5,
        },
    }


def test_hand_checkable_clustered_estimand() -> None:
    result = clustered.analyze_model(_toy_model_report(), repetitions=1_000, seed=7)

    assert result["market_ids"] == ["market_a", "market_b"]
    assert result["n_markets"] == 2
    assert result["direction_recovered_from_cf_id_records"] == 1
    assert result["signed_revision"]["pro_H1"]["estimate"] == pytest.approx(0.2)
    assert result["signed_revision"]["anti_H1"]["estimate"] == pytest.approx(-0.3)
    assert result["absolute_revision"]["directional"]["estimate"] == pytest.approx(0.25)
    assert result["absolute_revision"]["orthogonal"]["estimate"] == pytest.approx(0.02)
    assert result["sensitivity_ratio"]["estimate"] == pytest.approx(12.5)


def test_frozen_artifact_records_provenance_and_paper_values() -> None:
    artifact = json.loads(FROZEN_UNCERTAINTY.read_text(encoding="utf-8"))

    assert artifact["source_report_sha256"] == clustered._sha256(FROZEN_REPORT)
    assert artifact["bootstrap"] == {
        "cluster_unit": "market task_id",
        "repetitions": 20_000,
        "rng": "numpy.random.Generator(PCG64)",
        "seed": 20_260_822,
        "interval": "percentile 95% (2.5th, 97.5th percentiles)",
        "resampling": (
            "Sample markets with replacement; retain every packet and "
            "repeated initial-run update within each selected market."
        ),
        "estimand": (
            "Record-pooled mean within condition; sensitivity is the pooled "
            "directional absolute mean divided by the pooled orthogonal "
            "absolute mean, recomputed within each bootstrap replicate."
        ),
    }
    claude = artifact["per_model"]["claude-opus-4-8"]
    assert claude["n_markets"] == 100
    assert claude["sensitivity_ratio"] == pytest.approx(
        {
            "estimate": 24.03626290200292,
            "ci_low": 17.21112195569617,
            "ci_high": 36.324802728462835,
        }
    )
    assert set(artifact["per_model"]) == {
        model
        for model in json.loads(FROZEN_REPORT.read_text())["models"]
        if model not in clustered.EXCLUDED_MODELS
    }


def test_rebuilding_frozen_report_reproduces_point_estimates() -> None:
    rebuilt = clustered.build_report(
        FROZEN_REPORT, repetitions=1_000, seed=clustered.DEFAULT_SEED
    )
    frozen = json.loads(FROZEN_UNCERTAINTY.read_text(encoding="utf-8"))

    assert rebuilt["source_report_sha256"] == frozen["source_report_sha256"]
    for model, expected in frozen["per_model"].items():
        observed = rebuilt["per_model"][model]
        for direction in clustered.DIRECTIONS:
            assert observed["signed_revision"][direction]["estimate"] == pytest.approx(
                expected["signed_revision"][direction]["estimate"]
            )
            assert (
                observed["absolute_revision"][direction]["n_records"]
                == expected["absolute_revision"][direction]["n_records"]
            )
        assert observed["sensitivity_ratio"]["estimate"] == pytest.approx(
            expected["sensitivity_ratio"]["estimate"]
        )
