import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.three_city_c2_v7 import (
    CONTEXT_CONDITIONS,
    CONTEXT_SENSITIVITY_WEIGHTS,
    HIGH_RESPONSE,
    HIGH_TYPE,
    LOW_RESPONSE,
    ORTHOGONAL_CONTEXTS,
    PREFIX_LADDER,
    REFERENCE_CONTEXTS,
    STARTING_POLL,
    TARGET_CONTEXTS,
    balanced_triplets,
    compute_empirical_hierarchical,
    compute_target_only,
    estimate_response,
    generate_triplet,
    gold_expected_poll,
    infer_context_mechanism,
    infer_public_context_match,
    target_context,
)
from eval.build_three_city_c2_v7_tasks import (
    HINT_SENTENCE,
    build_prompt,
    build_records,
    strip_hint,
    validate_prompt,
)
from eval.run_three_city_c2_v7_confirmatory import _response_state


def test_calibration_and_prefixes_are_frozen():
    assert HIGH_RESPONSE == 0.90
    assert LOW_RESPONSE == 0.30
    assert PREFIX_LADDER == (0, 1, 2, 4)
    assert CONTEXT_CONDITIONS == ("relevant", "none", "orthogonal")


def test_blind_and_hint_differ_by_exactly_one_sentence():
    triplet = generate_triplet(seed=70_000)
    for condition in CONTEXT_CONDITIONS:
        for k in PREFIX_LADDER:
            blind = build_prompt(
                triplet,
                k=k,
                condition=condition,
                arm="blind",
            )
            hint = build_prompt(
                triplet,
                k=k,
                condition=condition,
                arm="hint",
            )
            assert HINT_SENTENCE not in blind
            assert hint.count(HINT_SENTENCE) == 1
            assert strip_hint(hint) == blind
            validate_prompt(blind, arm="blind")
            validate_prompt(hint, arm="hint")


def test_common_noise_reasoning_and_schema_are_identical():
    triplet = generate_triplet(seed=70_001)
    blind = build_prompt(
        triplet,
        k=2,
        condition="relevant",
        arm="blind",
    )
    hint = build_prompt(
        triplet,
        k=2,
        condition="relevant",
        arm="hint",
    )
    for phrase in (
        "vary from case to case",
        "do not provide step-by-step working",
        '"rationale": "one short sentence"',
        '"predicted_poll": <number from 0 to 100>',
    ):
        assert phrase in blind
        assert phrase in hint


def test_relevant_context_is_truthful_and_controls_are_clean():
    for triplet in balanced_triplets(120):
        family = triplet["context_family"]
        target_type = triplet["target"]["city_type"]
        assert target_context(triplet, "relevant") == (
            TARGET_CONTEXTS[target_type][family]
        )
        assert target_context(triplet, "orthogonal") == (
            ORTHOGONAL_CONTEXTS[family]
        )
        assert target_context(triplet, "none") == ""


def test_public_context_match_uses_only_displayed_background_text():
    for triplet in balanced_triplets(120):
        assert infer_context_mechanism(
            triplet["target_context_relevant"]
        ) == triplet["target"]["city_type"]
        assert infer_public_context_match(triplet) == triplet["target_matches"]

        contradicted_key = dict(triplet)
        contradicted_key["target_matches"] = (
            "B" if triplet["target_matches"] == "A" else "A"
        )
        assert infer_public_context_match(contradicted_key) == (
            triplet["target_matches"]
        )
        assert compute_empirical_hierarchical(
            contradicted_key,
            2,
            condition="relevant",
        ) == compute_empirical_hierarchical(
            triplet,
            2,
            condition="relevant",
        )


def test_target_only_is_deliberately_the_easy_city_c_average():
    triplet = generate_triplet(seed=70_002)
    for k in PREFIX_LADDER[1:]:
        target_only = compute_target_only(triplet, k)
        assert target_only["predicted_poll"] == pytest.approx(
            statistics.mean(triplet["target"]["ending_polls"][:k])
        )
        assert target_only["response_hat"] == pytest.approx(
            estimate_response(triplet["target"], k),
            abs=1e-6,
        )


def test_public_hierarchical_reference_beats_target_only_when_sparse():
    triplets = list(balanced_triplets(120))
    for condition in CONTEXT_CONDITIONS:
        for k in (1, 2):
            target_mae = statistics.mean(
                abs(
                    compute_target_only(triplet, k)["predicted_poll"]
                    - gold_expected_poll(triplet, k)
                )
                for triplet in triplets
            )
            hierarchical_mae = statistics.mean(
                abs(
                    compute_empirical_hierarchical(
                        triplet,
                        k,
                        condition=condition,
                    )["predicted_poll"]
                    - gold_expected_poll(triplet, k)
                )
                for triplet in triplets
            )
            assert hierarchical_mae < target_mae


def test_relevant_context_is_one_transparent_pseudo_case():
    triplet = generate_triplet(
        seed=70_003,
        target_type=HIGH_TYPE,
        high_reference="A",
    )
    no_context = compute_empirical_hierarchical(
        triplet,
        0,
        condition="none",
    )
    orthogonal = compute_empirical_hierarchical(
        triplet,
        0,
        condition="orthogonal",
    )
    relevant = compute_empirical_hierarchical(
        triplet,
        0,
        condition="relevant",
    )
    assert no_context["predicted_poll"] == orthogonal["predicted_poll"]
    assert relevant["context_pseudo_cases"] == 1.0
    assert relevant["p_match_a"] > 0.5


def test_reference_samples_are_representative_but_not_constant():
    triplet = generate_triplet(seed=70_004)
    for name in ("reference_a", "reference_b"):
        city = triplet[name]
        assert statistics.pstdev(city["ending_polls"]) > 0
        assert estimate_response(city) == pytest.approx(
            city["response"],
            abs=0.002,
        )


def test_context_templates_are_length_matched():
    relevant = [
        len(text.split())
        for texts in TARGET_CONTEXTS.values()
        for text in texts
    ]
    orthogonal = [len(text.split()) for text in ORTHOGONAL_CONTEXTS]
    assert max(relevant + orthogonal) - min(relevant + orthogonal) <= 2
    assert abs(statistics.mean(relevant) - statistics.mean(orthogonal)) <= 1
    reference = [
        len(text.split())
        for texts in REFERENCE_CONTEXTS.values()
        for text in texts
    ]
    assert max(reference) - min(reference) <= 3


def test_only_revealed_target_prefix_appears():
    triplet = generate_triplet(seed=70_005)
    polls = triplet["target"]["ending_polls"]
    for k in PREFIX_LADDER:
        prompt = build_prompt(
            triplet,
            k=k,
            condition="none",
            arm="blind",
        )
        city_c = prompt.split("CITY C", 1)[1]
        rows = [
            line
            for line in city_c.splitlines()
            if line.startswith("|") and line.split("|")[1].strip().isdigit()
        ]
        assert len(rows) == k
        for poll in polls[:k]:
            assert f"{poll:.1f}" in city_c
        for poll in polls[k:]:
            assert f"{poll:.1f}" not in city_c


def test_model_record_is_strictly_separated_from_answer_key():
    triplet = generate_triplet(seed=70_006)
    records, key = build_records(
        triplet,
        episode_index=6,
        k=2,
        condition="relevant",
    )
    for record in records.values():
        assert set(record) == {"task_id", "prompt", "prompt_sha256"}
        serialized = json.dumps(record)
        assert triplet["target"]["city_type"] not in serialized
        assert "target_matches" not in serialized
    assert set(key) >= {"task_id", "gold", "baselines", "condition"}
    sensitivity = key["baselines"][
        "empirical_hierarchical_context_sensitivity"
    ]
    assert set(sensitivity) == {
        f"{weight:g}" for weight in CONTEXT_SENSITIVITY_WEIGHTS
    }
    assert sensitivity["1"] == key["baselines"]["empirical_hierarchical"]
    assert records["blind"]["task_id"] == records["hint"]["task_id"] == key["task_id"]


def test_non_disclosure_guard_rejects_structural_instructions():
    disclosures = (
        "There are two types of city.",
        "City C matches City A.",
        "Estimate the response gain.",
        "Use a 50/50 prior.",
        "The contextual mechanism is known.",
    )
    for disclosure in disclosures:
        with pytest.raises(ValueError):
            validate_prompt(disclosure, arm="blind")


def test_gold_is_the_noiseless_expected_poll():
    triplet = generate_triplet(seed=70_007)
    for k in PREFIX_LADDER:
        assert gold_expected_poll(triplet, k) == pytest.approx(
            STARTING_POLL
            + triplet["target"]["response"] * triplet["test_news"]
        )


def test_runner_retries_transport_failures_but_not_parse_failures(tmp_path):
    path = tmp_path / "responses.jsonl"
    records = (
        {
            "arm": "blind",
            "task_id": "transport",
            "predicted_poll": None,
            "response_received": False,
        },
        {
            "arm": "blind",
            "task_id": "unparseable",
            "predicted_poll": None,
            "response_received": True,
        },
        {
            "arm": "hint",
            "task_id": "parsed",
            "predicted_poll": 54.0,
            "response_received": True,
        },
    )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    terminal, parsed = _response_state(path)
    assert terminal == {
        ("blind", "unparseable"),
        ("hint", "parsed"),
    }
    assert parsed == {("hint", "parsed")}
