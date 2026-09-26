"""Checks for the Coin City reference-selection analysis.

The estimand is a partial correlation, so the tests target the two ways it could
be wrong: the partial-correlation implementation itself, and the mapping from an
arm to the reference city its City C label names.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_coin_city_reference_selection import (  # noqa: E402
    CUED_ARMS,
    PRIMARY_ARMS,
    PRIMARY_MODELS,
    SYMBOL_MODELS,
    bootstrap_partial,
    episode_vectors,
    partial_correlation,
    read_jsonl,
    reference_noise_floor,
    through_origin_slope,
)

DESIGN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4" / "design"
RESULTS = (
    ROOT
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "reference_selection_results.json"
)


@pytest.fixture(scope="module")
def episodes() -> dict[int, dict]:
    return {row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")}


@pytest.fixture(scope="module")
def scores() -> dict[str, dict]:
    return {row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")}


def test_through_origin_slope_recovers_a_noiseless_fit():
    rows = [
        {"starting_poll": 50.0, "net_news": 8, "ending_poll": 50.0 + 0.9 * 8},
        {"starting_poll": 50.0, "net_news": -8, "ending_poll": 50.0 - 0.9 * 8},
    ]
    assert through_origin_slope(rows) == pytest.approx(0.9)
    assert through_origin_slope([]) == 0.0


def test_partial_correlation_removes_the_control():
    rng = np.random.default_rng(0)
    control = np.repeat([0.0, 1.0], 200)
    noise = rng.normal(size=400)
    # Both variables are driven only by the control: no partial association.
    left = 3.0 * control + rng.normal(size=400)
    right = 3.0 * control + rng.normal(size=400)
    assert abs(partial_correlation(left, right, control)) < 0.15
    # Now give them a shared within-group component; the partial must find it.
    left = 3.0 * control + noise
    right = 3.0 * control + noise + 0.1 * rng.normal(size=400)
    assert partial_correlation(left, right, control) > 0.9
    # A raw correlation would be high in both cases.
    assert np.corrcoef(3.0 * control + rng.normal(size=400), control)[0, 1] > 0.5


def test_partial_correlation_is_degenerate_safe():
    control = np.array([0.0, 1.0, 0.0, 1.0])
    assert partial_correlation(np.ones(4), np.arange(4.0), control) is None
    assert partial_correlation(np.array([1.0]), np.array([1.0]), np.array([0.0])) is None


def test_bootstrap_partial_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    control = np.repeat([0.0, 1.0], 60)
    shared = rng.normal(size=120)
    left = control + shared
    right = control + shared + 0.5 * rng.normal(size=120)
    first = bootstrap_partial(left, right, control, seed=7, draws=200)
    second = bootstrap_partial(left, right, control, seed=7, draws=200)
    assert first == second
    low, high = first["ci_95"]
    assert low <= first["partial_r"] <= high


def test_cue_arms_select_the_named_reference(episodes, scores):
    """The correct and inverted arms must name opposite reference cities."""
    task_ids = sorted(
        task_id for task_id, row in scores.items() if int(row["c_cases"]) == 0
    )[:40]
    correct = episode_vectors(episodes, scores, task_ids, "abc_context")
    wrong = episode_vectors(episodes, scores, task_ids, "abc_wrong_context")
    none = episode_vectors(episodes, scores, task_ids, "abc_no_context")

    # Inversion swaps which reference is named, and swaps the named regime.
    assert np.allclose(correct["cue_reference_ols"], wrong["other_reference_ols"])
    assert np.allclose(correct["other_reference_ols"], wrong["cue_reference_ols"])
    assert np.allclose(correct["named_regime"], 1.0 - wrong["named_regime"])
    # Without a label the arm falls back to the true regime, as the correct arm does.
    assert np.allclose(none["named_regime"], correct["named_regime"])
    assert np.allclose(none["named_regime"], correct["true_regime"])
    # The named reference really is the one whose city carries the named regime.
    for index, task_id in enumerate(task_ids):
        episode = episodes[scores[task_id]["episode"]]
        a_is_strong = episode["strong_reference_city"] == "A"
        named_strong = bool(correct["named_regime"][index])
        expected = episode["reference_a"] if a_is_strong == named_strong else episode["reference_b"]
        assert correct["cue_reference_ols"][index] == pytest.approx(
            through_origin_slope(expected)
        )


def test_arm_cue_table_matches_the_frozen_arm_names():
    assert CUED_ARMS["abc_context"] is True
    assert CUED_ARMS["abc_symbol_context"] is True
    assert CUED_ARMS["abc_wrong_context"] is False
    assert "abc_no_context" not in CUED_ARMS
    assert "baseline" not in CUED_ARMS


def test_reference_noise_floor_is_the_documented_gap(episodes):
    floor = reference_noise_floor(episodes)
    assert floor["episodes"] == 250
    # The regime nearly determines the slope, but a four-case fit does not.
    assert floor["regime_vs_true_slope_r"] > 0.99
    assert 0.6 < floor["matched_reference_ols_vs_true_slope_r"] < 0.7
    assert floor["matched_reference_ols_within_regime_sd"] > 10 * floor[
        "generator_within_regime_sd"
    ]


@pytest.mark.skipif(not RESULTS.exists(), reason="analysis output not built")
def test_frozen_results_show_cue_conditioned_selection():
    result = json.loads(RESULTS.read_text())
    assert result["model_calls"] == 0
    assert set(result["primary"]) == set(PRIMARY_MODELS)
    assert set(result["symbol"]) <= set(SYMBOL_MODELS)

    for model in PRIMARY_MODELS:
        arms = result["primary"][model]["arms"]
        assert set(arms) == set(PRIMARY_ARMS)
        for arm in ("abc_context", "abc_wrong_context"):
            cell = arms[arm]["0"]
            named, other = cell["cue_reference"], cell["other_reference"]
            # The named reference loads; the other one does not.
            assert named["ci_95"][0] > 0, (model, arm)
            assert other["ci_95"][0] < 0 < other["ci_95"][1], (model, arm)
            assert named["partial_r"] > other["partial_r"] + 0.3, (model, arm)
        # With no label the two references load about equally: the pooling control.
        pooled = arms["abc_no_context"]["0"]
        gap = abs(
            pooled["regime_matched_reference"]["partial_r"]
            - pooled["regime_mismatched_reference"]["partial_r"]
        )
        assert gap < 0.2, (model, gap)
        # With no reference cases displayed there is nothing to track.
        empty = arms["baseline"]["0"]
        assert abs(empty["regime_matched_reference"]["partial_r"]) < 0.2, model


@pytest.mark.skipif(not RESULTS.exists(), reason="analysis output not built")
def test_symbol_arm_agrees_with_the_headline_discrimination_contrast():
    """Both statistics must name the same three non-recovering systems."""
    result = json.loads(RESULTS.read_text())
    comparison = json.loads(
        (
            ROOT
            / "data"
            / "coin_city_stable_relationship_claude_n250_v4"
            / "analysis"
            / "symbol_context_model_comparison_20260824.json"
        ).read_text()
    )
    selects, discriminates = set(), set()
    for model, entry in result["symbol"].items():
        if entry["arms"]["abc_symbol_context"]["0"]["cue_reference"]["ci_95"][0] > 0:
            selects.add(model)
        curve = comparison["models"][model]["curves"]["0"]
        if curve["paired_discrimination"]["symbol_minus_no_context_rho"]["ci_95"][0] > 0:
            discriminates.add(model)
    assert selects == discriminates
    assert {"Qwen3-4B-Instruct-2507", "Qwen3-8B", "Llama-3.1-8B-Instruct"} == (
        set(result["symbol"]) - selects
    )
