"""Tests for the two-reference-cities → third-city redesign."""

import json
import copy
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from engine.three_city import (
    PROBE_SHOCKS,
    compute_clairvoyant_ceiling,
    compute_frequentist_baseline,
    compute_hierarchical_bayes,
    compute_naive_baseline,
    compute_pooled_baseline,
    counterfactual_gold,
    generate_triplet,
    position_score,
    type_score,
)
from run_three_city import build_prompt, parse_reply


def test_triplet_has_two_types_and_target_matches_one():
    triplet = generate_triplet(seed=10)
    gains = {
        triplet["reference_a"]["gain"],
        triplet["reference_b"]["gain"],
    }
    assert gains == {0.25, 1.0}
    assert triplet["target_matches"] in {"A", "B"}
    matched = triplet[f"reference_{triplet['target_matches'].lower()}"]["gain"]
    assert triplet["target"]["gain"] == matched


def test_labels_and_target_match_are_randomised():
    triplets = [generate_triplet(seed=seed) for seed in range(100)]
    assert {triplet["strong_reference"] for triplet in triplets} == {"A", "B"}
    assert {triplet["target_matches"] for triplet in triplets} == {"A", "B"}


def test_generation_is_reproducible():
    first = generate_triplet(seed=42)
    second = generate_triplet(seed=42)
    assert first == second


def test_fully_sticky_zero_gain_variant():
    triplet = generate_triplet(seed=7, weak_gain=0.0)
    gains = {
        triplet["reference_a"]["gain"],
        triplet["reference_b"]["gain"],
    }
    assert gains == {0.0, 1.0}
    result = compute_hierarchical_bayes(triplet, k=1)
    assert 0.0 <= result["gain_hat"] <= 1.0


def test_oracle_starts_at_fifty_fifty():
    triplet = generate_triplet(seed=4)
    result = compute_hierarchical_bayes(triplet, k=0)
    assert result["p_match_a"] == 0.5
    assert result["p_match_b"] == 0.5


def test_diagnostic_poll_quickly_identifies_the_right_reference():
    correct_probabilities = []
    for seed in range(100):
        triplet = generate_triplet(seed=seed, target_evidence="diagnostic")
        result = compute_hierarchical_bayes(triplet, k=1)
        correct_probabilities.append(
            type_score(triplet, result["p_match_a"])["p_correct_type"]
        )
    assert statistics.mean(correct_probabilities) > 0.85


def test_hierarchical_oracle_beats_permanent_pooling():
    oracle_errors = []
    pooled_errors = []
    for seed in range(100):
        triplet = generate_triplet(seed=seed)
        oracle = compute_hierarchical_bayes(triplet, k=2)
        pooled = compute_pooled_baseline(triplet, k=2)
        oracle_errors.append(
            position_score(triplet, oracle["gain_hat"])["gain_error"]
        )
        pooled_errors.append(
            position_score(triplet, pooled["gain_hat"])["gain_error"]
        )
    assert statistics.mean(oracle_errors) < statistics.mean(pooled_errors)


def test_naive_gain_is_the_fixed_population_midpoint():
    triplet = generate_triplet(seed=8)
    assert compute_naive_baseline(triplet, k=0)["gain_hat"] == 0.625
    assert compute_naive_baseline(triplet, k=5)["gain_hat"] == 0.625


def test_frequentist_uses_only_target_city_data():
    triplet = generate_triplet(seed=11)
    original = compute_frequentist_baseline(triplet, k=3)
    changed_references = copy.deepcopy(triplet)
    changed_references["reference_a"]["news"] = [100] * 6
    changed_references["reference_a"]["polls"] = [0] * 6
    changed_references["reference_b"]["news"] = [-100] * 6
    changed_references["reference_b"]["polls"] = [100] * 6
    changed = compute_frequentist_baseline(changed_references, k=3)
    assert changed == original
    assert compute_frequentist_baseline(triplet, k=0)["available"] is False


def test_clairvoyant_ceiling_matches_counterfactual_gold():
    triplet = generate_triplet(seed=12)
    ceiling = compute_clairvoyant_ceiling(triplet, k=2)
    assert ceiling["gain_hat"] == triplet["target"]["gain"]
    assert ceiling["forecasts"] == counterfactual_gold(triplet, k=2)


def test_counterfactual_gold_has_true_gain_as_slope():
    triplet = generate_triplet(seed=3)
    gold = counterfactual_gold(triplet, k=2)
    low, high = str(PROBE_SHOCKS[0]), str(PROBE_SHOCKS[-1])
    slope = (
        gold[high] - gold[low]
    ) / (
        PROBE_SHOCKS[-1] - PROBE_SHOCKS[0]
    )
    assert abs(slope - triplet["target"]["gain"]) < 1e-6


def test_prompt_contains_all_three_cities_and_only_target_prefix():
    triplet = generate_triplet(seed=2)
    prompt = build_prompt(triplet, k=1)
    assert "REFERENCE CITY A" in prompt
    assert "REFERENCE CITY B" in prompt
    assert "TARGET CITY C" in prompt
    first_target_row = (
        f"| {1:>4} | {triplet['target']['news'][0]:>+8d} | "
        f"{triplet['target']['polls'][0]:>4} |"
    )
    second_target_row = (
        f"| {2:>4} | {triplet['target']['news'][1]:>+8d} | "
        f"{triplet['target']['polls'][1]:>4} |"
    )
    assert first_target_row in prompt
    assert second_target_row not in prompt


def test_reply_parser_and_behavioural_schema():
    reply = {
        "prob_matches_a": 80,
        "points_per_news": 0.75,
        "predictions": [
            {"news": -10, "poll": 42},
            {"news": -5, "poll": 46},
            {"news": 5, "poll": 54},
            {"news": 10, "poll": 58},
        ],
    }
    parsed = parse_reply(f"```json\n{json.dumps(reply)}\n```")
    assert parsed["prob_matches_a"] == 0.8
    assert parsed["points_per_news"] == 0.75
    assert len(parsed["predictions"]) == 4
