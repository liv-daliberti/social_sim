"""Design and fairness gates for the structure-blind C2 experiment."""

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from engine.three_city_c2 import (
    HIGH_TYPE,
    LOW_TYPE,
    behavioral_gain,
    balanced_triplets,
    compute_bayes,
    compute_clairvoyant,
    compute_frequentist,
    compute_naive,
    compute_pooled_exemplar,
    compute_two_prototype_exemplar,
    generate_triplet,
    gold_expected_poll,
    reference_ols_gain,
    score_type,
    target_context,
)
from build_three_city_c2_tasks import (
    build_prompt,
    build_records,
    validate_structure_blind_prompt,
)


def test_balanced_registry_crosses_type_label_sign_and_context_family():
    triplets = list(balanced_triplets(48))
    assert sum(t["target"]["city_type"] == HIGH_TYPE for t in triplets) == 24
    assert sum(t["high_reference"] == "A" for t in triplets) == 24
    assert {1 if t["target"]["news"][0] > 0 else -1 for t in triplets} == {-1, 1}
    assert {t["context_family"] for t in triplets} == {0, 1, 2}


def test_generation_is_reproducible():
    first = generate_triplet(seed=41)
    second = generate_triplet(seed=41)
    assert first == second


def test_reference_examples_make_both_patterns_estimable():
    errors = []
    for triplet in balanced_triplets(200):
        for label in ("reference_a", "reference_b"):
            city = triplet[label]
            errors.append(abs(reference_ols_gain(city) - city["gain"]))
    assert statistics.mean(errors) < 0.04


def test_first_target_observation_is_weak_and_second_is_diagnostic():
    triplets = list(balanced_triplets(500))
    correct_by_k = {}
    for k in (1, 2):
        probabilities = []
        for triplet in triplets:
            result = compute_bayes(
                triplet,
                k,
                condition="none",
            )
            probabilities.append(
                score_type(triplet, result["p_match_a"])["p_correct_type"]
            )
        correct_by_k[k] = statistics.mean(probabilities)
    assert 0.50 < correct_by_k[1] < 0.65
    assert correct_by_k[2] > 0.94


def test_diagnostic_evidence_overrides_a_misleading_context_cue():
    triplets = list(balanced_triplets(500))
    correct_by_k = {}
    for k in (0, 2):
        probabilities = []
        for triplet in triplets:
            misleading = (
                "cue_low"
                if triplet["target"]["city_type"] == HIGH_TYPE
                else "cue_high"
            )
            result = compute_bayes(
                triplet,
                k,
                condition=misleading,
            )
            probabilities.append(
                score_type(triplet, result["p_match_a"])["p_correct_type"]
            )
        correct_by_k[k] = statistics.mean(probabilities)
    assert correct_by_k[0] == 0.2
    assert correct_by_k[2] > 0.93


def test_orthogonal_context_does_not_change_oracle_forecast():
    for triplet in balanced_triplets(20):
        for k in range(4):
            none = compute_bayes(triplet, k, condition="none")
            orthogonal = compute_bayes(
                triplet,
                k,
                condition="orthogonal",
            )
            assert none["predicted_poll"] == orthogonal["predicted_poll"]
            assert none["p_match_a"] == orthogonal["p_match_a"]


def test_context_swap_holds_numbers_fixed_and_moves_context_oracle():
    triplet = generate_triplet(seed=5)
    high = compute_bayes(triplet, 0, condition="cue_high")
    low = compute_bayes(triplet, 0, condition="cue_low")
    assert target_context(triplet, "cue_high") != target_context(
        triplet,
        "cue_low",
    )
    expected_direction = 1 if triplet["test_news"] > 0 else -1
    assert (
        high["predicted_poll"] - low["predicted_poll"]
    ) * expected_direction > 0


def test_prompt_does_not_disclose_the_evaluator_structure():
    triplet = generate_triplet(seed=9)
    for condition in ("none", "orthogonal", "cue_high", "cue_low"):
        for k in range(4):
            prompt = build_prompt(
                triplet,
                k=k,
                condition=condition,
            )
            validate_structure_blind_prompt(prompt)
            lowered = prompt.lower()
            assert "city a" in lowered
            assert "city b" in lowered
            assert "city c" in lowered
            assert "predicted_poll" in prompt
            assert "prob_matches" not in prompt
            assert "points_per_news" not in prompt


def test_non_disclosure_guard_rejects_structural_instructions():
    disclosures = (
        "There are two types of city.",
        "City C matches City A.",
        "The response gain is hidden.",
        "Use a 50/50 prior.",
        "The contextual cue is reliable.",
        "Both cities follow the same response pattern.",
    )
    for disclosure in disclosures:
        with pytest.raises(ValueError):
            validate_structure_blind_prompt(disclosure)


def test_model_facing_record_is_separated_from_evaluator_key():
    triplet = generate_triplet(seed=11)
    model_record, evaluator_record = build_records(
        triplet,
        episode_index=11,
        k=1,
        condition="cue_high",
    )
    assert set(model_record) == {"task_id", "prompt", "prompt_sha256"}
    assert set(evaluator_record) >= {
        "task_id",
        "condition",
        "gold",
        "baselines",
    }
    assert model_record["task_id"] == evaluator_record["task_id"]
    serialized = json.dumps(model_record)
    assert "cue_high" not in serialized
    assert triplet["target"]["city_type"] not in serialized


def test_prompt_variants_change_only_target_background():
    triplet = generate_triplet(seed=10)
    none = build_prompt(triplet, k=1, condition="none")
    high = build_prompt(triplet, k=1, condition="cue_high")
    low = build_prompt(triplet, k=1, condition="cue_low")
    target_row = (
        f"| {1:>4} | {triplet['target']['news'][0]:>+8d} | "
        f"{triplet['target']['polls'][0]:>5.1f} |"
    )
    assert target_row in none and target_row in high and target_row in low
    assert triplet["target_context_high"] in high
    assert triplet["target_context_low"] in low


def test_naive_repeats_current_poll_and_frequentist_has_no_k0_estimate():
    triplet = generate_triplet(seed=12)
    assert compute_naive(triplet, 0)["predicted_poll"] == 50.0
    assert compute_frequentist(triplet, 0)["available"] is False
    for k in (1, 2, 3):
        assert compute_naive(triplet, k)["predicted_poll"] == triplet[
            "target"
        ]["polls"][k - 1]


def test_observation_matched_exemplar_baselines_are_finite():
    for triplet in balanced_triplets(50):
        for k in range(4):
            pooled = compute_pooled_exemplar(triplet, k)
            prototype = compute_two_prototype_exemplar(triplet, k)
            assert 0 <= pooled["predicted_poll"] <= 100
            assert 0 <= prototype["predicted_poll"] <= 100
            assert 0 <= prototype["p_match_a"] <= 1


def test_clairvoyant_forecast_defines_gold_and_behavioral_gain():
    triplet = generate_triplet(seed=13)
    for k in range(4):
        result = compute_clairvoyant(triplet, k)
        assert result["predicted_poll"] == gold_expected_poll(triplet, k)
        implied = behavioral_gain(
            triplet,
            k,
            result["predicted_poll"],
        )
        assert abs(implied - triplet["target"]["gain"]) < 1e-9
