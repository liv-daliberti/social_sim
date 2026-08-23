"""Design gates for the minimal independent-case C2 v6 experiment."""

import json
import random
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from build_three_city_c2_v6_tasks import (
    build_prompt,
    build_records,
    validate_structure_blind_prompt,
)
from engine.three_city_c2_v6 import (
    HIGH_RESPONSE,
    CONTEXT_CONDITIONS,
    REFERENCE_CASES,
    REFERENCE_MAGNITUDES,
    TARGET_MAGNITUDE,
    news_schedule,
    HIGH_TYPE,
    LOW_TYPE,
    NEWS_VALUE,
    ORTHOGONAL_CONTEXTS,
    PREFIX_LADDER,
    REFERENCE_CONTEXTS,
    STARTING_POLL,
    TARGET_CASES,
    TARGET_CONTEXTS,
    balanced_triplets,
    behavioral_response,
    compute_bayes,
    compute_frequentist,
    compute_pooled_exemplar,
    estimate_response,
    generate_triplet,
    gold_expected_poll,
    score_type,
    target_context,
)


def test_cases_are_independent_with_one_fixed_start_and_varied_news():
    triplet = generate_triplet(seed=20)
    for label in ("reference_a", "reference_b", "target"):
        city = triplet[label]
        assert city["starting_poll"] == STARTING_POLL
        assert len(city["ending_polls"]) == len(city["news"])
        # v6: one magnitude per city, taking both signs, summing to zero.
        assert len(set(city["news"])) == 2
        assert sum(city["news"]) == 0
    # The forecast case is still a single fixed value, so the readout is
    # unchanged from v3.
    assert triplet["test_news"] == NEWS_VALUE


def test_each_city_uses_one_magnitude_with_both_signs():
    for triplet in balanced_triplets(24):
        for label in ("reference_a", "reference_b", "target"):
            news = triplet[label]["news"]
            magnitude = abs(news[0])
            assert set(news) == {magnitude, -magnitude}
            assert sum(news) == 0, "the schedule must sum to zero"
            # Alternating, so every even-length prefix is sign-balanced and the
            # response stays estimable at every rung of the ladder.
            assert all(a == -b for a, b in zip(news, news[1:]))


def test_three_cities_use_three_different_magnitudes():
    """Raw poll levels must not be comparable across cities without
    normalising by each city's own news magnitude."""
    for triplet in balanced_triplets(24):
        magnitudes = {
            abs(triplet[label]["news"][0])
            for label in ("reference_a", "reference_b", "target")
        }
        assert len(magnitudes) == 3
        assert triplet["test_news"] not in magnitudes


def test_magnitude_carries_no_information_about_response_type():
    high_magnitudes = []
    for triplet in balanced_triplets(120):
        for label in ("reference_a", "reference_b"):
            city = triplet[label]
            if city["response"] == HIGH_RESPONSE:
                high_magnitudes.append(abs(city["news"][0]))
    counts = [high_magnitudes.count(m) for m in REFERENCE_MAGNITUDES]
    assert min(counts) > 0.35 * sum(counts), (
        "magnitude predicts response type; it must be assigned independently"
    )


def test_the_response_is_recoverable_from_two_group_means():
    """v6's core claim: the estimator is two means and a division, not a
    regression. This is what keeps the arithmetic cost from swamping the
    effect the experiment measures, as it did in v5."""
    errors = []
    for triplet in balanced_triplets(120):
        city = triplet["target"]
        magnitude = abs(city["news"][0])
        plus = [p for d, p in zip(city["news"], city["ending_polls"]) if d > 0]
        minus = [p for d, p in zip(city["news"], city["ending_polls"]) if d < 0]
        estimate = (
            statistics.mean(plus) - statistics.mean(minus)
        ) / (2 * magnitude)
        errors.append(abs(estimate - city["response"]))
    assert statistics.mean(errors) < 0.20


def test_averaging_the_column_no_longer_estimates_the_response():
    """The shortcut v3 permitted must be genuinely broken, not just discouraged."""
    implied = []
    for triplet in balanced_triplets(240):
        target = triplet["target"]
        mean_poll = statistics.mean(target["ending_polls"])
        implied.append(
            (mean_poll - target["starting_poll"]) / triplet["test_news"]
        )
    # True responses are 0.30 and 0.90; a column average implies roughly nothing.
    assert abs(statistics.mean(implied)) < 0.15


def test_generation_is_reproducible():
    assert generate_triplet(seed=21) == generate_triplet(seed=21)


def test_balanced_registry_crosses_type_label_and_context_family():
    triplets = list(balanced_triplets(120))
    assert sum(t["target"]["city_type"] == HIGH_TYPE for t in triplets) == 60
    assert sum(t["high_reference"] == "A" for t in triplets) == 60
    assert {t["context_family"] for t in triplets} == {0, 1, 2}
    assert {
        family: sum(t["context_family"] == family for t in triplets)
        for family in (0, 1, 2)
    } == {0: 40, 1: 40, 2: 40}


def test_reference_examples_estimate_distinct_responses():
    errors = []
    separations = []
    for triplet in balanced_triplets(500):
        response_a = estimate_response(triplet["reference_a"])
        response_b = estimate_response(triplet["reference_b"])
        errors.extend(
            (
                abs(response_a - triplet["reference_a"]["response"]),
                abs(response_b - triplet["reference_b"]["response"]),
            )
        )
        separations.append(abs(response_a - response_b))
    # Reference cities are mean-centred, so their estimates are near-exact and
    # the recovered gap sits on the true 0.60 rather than v2's 0.75.
    assert statistics.mean(errors) < 0.01
    assert statistics.mean(separations) > 0.55


def test_target_information_increases_gradually_along_the_ladder():
    triplets = list(balanced_triplets(1000))
    correct = {}
    for k in PREFIX_LADDER:
        correct[k] = statistics.mean(
            score_type(
                triplet,
                compute_bayes(
                    triplet,
                    k,
                    condition="none",
                )["p_match_a"],
            )["p_correct_type"]
            for triplet in triplets
        )
    assert correct[0] == 0.5
    assert 0.55 < correct[1] < 0.75
    assert all(
        correct[a] < correct[b]
        for a, b in zip(PREFIX_LADDER, PREFIX_LADDER[1:])
    )
    # v3 is calibrated so the ladder top is informative but unsettled, and the
    # upper half still moves. v2 failed both: 0.93 by k=3, 0.997 by k=8.
    assert 0.85 < correct[TARGET_CASES] < 0.96
    assert correct[TARGET_CASES] - correct[3] >= 0.10


def test_ladder_top_overrides_a_misleading_context_on_average():
    triplets = list(balanced_triplets(1000))
    correct = {}
    for k in (0, 1, TARGET_CASES):
        values = []
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
            values.append(
                score_type(
                    triplet,
                    result["p_match_a"],
                )["p_correct_type"]
            )
        correct[k] = statistics.mean(values)
    assert correct[0] == 0.2
    assert correct[1] > 0.30
    assert correct[TARGET_CASES] > 0.78


def test_frequentist_improves_gradually_and_does_not_get_k0_information():
    triplets = list(balanced_triplets(1000))
    frequentist_mae = {}
    ceiling_mae = {}
    for k in PREFIX_LADDER:
        frequentist_errors = []
        ceiling_errors = []
        for triplet in triplets:
            gold = gold_expected_poll(triplet, k)
            frequentist = compute_frequentist(triplet, k)
            ceiling = compute_bayes(triplet, k, condition="none")
            if frequentist["predicted_poll"] is not None:
                frequentist_errors.append(
                    abs(frequentist["predicted_poll"] - gold)
                )
            ceiling_errors.append(abs(ceiling["predicted_poll"] - gold))
        frequentist_mae[k] = (
            statistics.mean(frequentist_errors)
            if frequentist_errors
            else None
        )
        ceiling_mae[k] = statistics.mean(ceiling_errors)
    assert frequentist_mae[0] is None
    assert all(
        frequentist_mae[a] > frequentist_mae[b]
        for a, b in zip(PREFIX_LADDER[1:], PREFIX_LADDER[2:])
    )
    assert frequentist_mae[1] > 2.0
    assert all(
        ceiling_mae[k] < frequentist_mae[k] for k in PREFIX_LADDER[1:]
    )


def test_ceiling_gap_narrows_but_stays_open_at_the_ladder_top():
    """A two-point prior collapses exponentially; a sample mean does not.

    The ladder therefore narrows the gap without closing it. This pins the
    interim status of TARGET_CASES = 8 so a later change to the anchor is a
    deliberate edit rather than a silent drift.
    """
    triplets = list(balanced_triplets(1000))
    gap = {}
    for k in (1, TARGET_CASES):
        gap[k] = statistics.mean(
            abs(
                compute_frequentist(triplet, k)["predicted_poll"]
                - gold_expected_poll(triplet, k)
            )
            - abs(
                compute_bayes(triplet, k, condition="none")["predicted_poll"]
                - gold_expected_poll(triplet, k)
            )
            for triplet in triplets
        )
    assert gap[1] > gap[TARGET_CASES] > 0.5


def test_gold_is_the_noiseless_response_and_defines_behavioral_response():
    triplet = generate_triplet(seed=22)
    for k in PREFIX_LADDER:
        gold = gold_expected_poll(triplet, k)
        assert gold == round(
            STARTING_POLL
            + triplet["target"]["response"] * triplet["test_news"],
            3,
        )
        assert behavioral_response(triplet, gold) == pytest.approx(
            triplet["target"]["response"]
        )


def test_pooled_exemplar_ignores_the_target_city():
    triplet = generate_triplet(seed=23)
    predictions = {
        compute_pooled_exemplar(triplet, k)["predicted_poll"]
        for k in PREFIX_LADDER
    }
    assert len(predictions) == 1


def test_context_templates_are_length_matched():
    relevant = [
        text
        for collection in (REFERENCE_CONTEXTS, TARGET_CONTEXTS)
        for texts in collection.values()
        for text in texts
    ]
    orthogonal_lengths = [len(text.split()) for text in ORTHOGONAL_CONTEXTS]
    assert max(len(text.split()) for text in relevant) - min(
        len(text.split()) for text in relevant
    ) <= 3
    assert max(orthogonal_lengths) - min(orthogonal_lengths) <= 1
    assert (
        abs(
            statistics.mean(len(text.split()) for text in relevant)
            - statistics.mean(orthogonal_lengths)
        )
        < 1.0
    )


def test_context_swap_changes_words_not_numbers():
    triplet = generate_triplet(seed=23)
    high = build_prompt(triplet, k=TARGET_CASES, condition="cue_high")
    low = build_prompt(triplet, k=TARGET_CASES, condition="cue_low")
    assert target_context(triplet, "cue_high") in high
    assert target_context(triplet, "cue_low") in low
    for poll in triplet["target"]["ending_polls"]:
        assert f"{poll:.1f}" in high
        assert f"{poll:.1f}" in low


def test_only_the_revealed_prefix_of_target_cases_appears():
    triplet = generate_triplet(seed=24)
    polls = triplet["target"]["ending_polls"]
    for k in PREFIX_LADDER:
        prompt = build_prompt(triplet, k=k, condition="none")
        table = prompt.split("CITY C", 1)[1]
        data_rows = [
            line
            for line in table.splitlines()
            if line.startswith("|") and line.split("|")[1].strip().isdigit()
        ]
        assert len(data_rows) == k
        for poll in polls[:k]:
            assert f"{poll:.1f}" in table
        # Deliberately not asserting that hidden polls are absent from the
        # rendered text: with one magnitude per city two different cases can
        # legitimately round to the same poll value, so that check tests
        # coincidence rather than correctness. The row count above is the real
        # guarantee that only k cases are shown.


def test_orthogonal_context_does_not_change_oracle_forecast():
    for triplet in balanced_triplets(20):
        for k in PREFIX_LADDER:
            none = compute_bayes(triplet, k, condition="none")
            orthogonal = compute_bayes(
                triplet,
                k,
                condition="orthogonal",
            )
            assert none["predicted_poll"] == orthogonal["predicted_poll"]


def test_prompt_is_explicit_about_sampling_but_blind_to_latent_structure():
    triplet = generate_triplet(seed=24)
    for condition in CONTEXT_CONDITIONS:
        for k in range(4):
            prompt = build_prompt(
                triplet,
                k=k,
                condition=condition,
            )
            validate_structure_blind_prompt(prompt)
            lowered = prompt.lower()
            assert "separate polling case" in lowered
            assert "not a time series" in lowered
            assert "poll at end of case" in lowered
            assert "city a" in lowered
            assert "city b" in lowered
            assert "city c" in lowered


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


def test_model_record_is_strictly_separated_from_evaluator_key():
    triplet = generate_triplet(seed=25)
    model_record, evaluator_record = build_records(
        triplet,
        episode_index=25,
        k=3,
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
