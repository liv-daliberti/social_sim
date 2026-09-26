import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.three_city_c2_v8 import (
    CASES_BY_PREFIX,
    CONTEXT_CONDITIONS,
    HIGH_TYPE,
    PREFIX_LADDER,
    PROMPT_ARMS,
    balanced_triplets,
    cases_for_prefix,
    compute_abc_shrinkage,
    compute_context_oracle,
    compute_target_only,
    gold_expected_poll,
)
from eval.build_three_city_c2_v8_tasks import (
    HINT_BLOCK,
    STRONG_HINT_BLOCK,
    build_prompt,
    build_records,
    strip_arm_block,
    validate_arm_prompt,
)
from eval.run_three_city_c2_v8_confirmatory import (
    _parse_reply,
    _response_state,
)


def _mae(triplets, method, k):
    return statistics.mean(
        abs(method(triplet, k)["predicted_poll"] - gold_expected_poll(triplet, k))
        for triplet in triplets
    )


def test_frozen_ladder_arms_and_contexts():
    assert PREFIX_LADDER == tuple(range(1, 6))
    assert CASES_BY_PREFIX == {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
    assert PROMPT_ARMS == ("blind", "hint", "strong_hint")
    assert CONTEXT_CONDITIONS == (
        "none",
        "orthogonal",
        "cue_high",
        "cue_low",
    )


def test_three_arms_differ_only_by_their_exact_frozen_blocks():
    triplet = next(iter(balanced_triplets(12, seed_offset=80_000)))
    for condition in CONTEXT_CONDITIONS:
        for k in PREFIX_LADDER:
            prompts = {
                arm: build_prompt(
                    triplet,
                    k=k,
                    condition=condition,
                    arm=arm,
                )
                for arm in PROMPT_ARMS
            }
            for arm, prompt in prompts.items():
                validate_arm_prompt(prompt, arm)
            assert HINT_BLOCK not in prompts["blind"]
            assert STRONG_HINT_BLOCK not in prompts["blind"]
            assert prompts["hint"].count(HINT_BLOCK) == 1
            assert prompts["strong_hint"].count(STRONG_HINT_BLOCK) == 1
            assert strip_arm_block(prompts["hint"], "hint") == prompts["blind"]
            assert (
                strip_arm_block(prompts["strong_hint"], "strong_hint")
                == prompts["blind"]
            )


def test_blind_prompt_does_not_disclose_evaluator_structure():
    triplet = next(iter(balanced_triplets(12, seed_offset=80_001)))
    prompt = build_prompt(
        triplet,
        k=2,
        condition="none",
        arm="blind",
    )
    lower = prompt.lower()
    for forbidden in (
        "two recurring response patterns",
        "city c follows one",
        "equally likely",
        "50/50",
        "sigma",
        "open_information",
        "buffered_information",
    ):
        assert forbidden not in lower
    assert "regional-profile index" in lower
    for required in (
        "separate polling case",
        "not a time series",
        "vary from case to case",
        "do not provide step-by-step working",
    ):
        assert required in lower


def test_strong_hint_discloses_continuous_pooling_but_not_answers():
    triplet = next(iter(balanced_triplets(12, seed_offset=80_002)))
    prompt = build_prompt(
        triplet,
        k=2,
        condition="none",
        arm="strong_hint",
    )
    assert STRONG_HINT_BLOCK in prompt
    lower = prompt.lower()
    for forbidden in (
        "1.00",
        "0.25",
        "50/50",
        "sigma",
        triplet["target"]["city_type"].lower(),
        triplet["target_matches"].lower() + " is the match",
    ):
        assert forbidden not in lower


def test_model_records_are_private_key_free_and_share_task_ids():
    triplet = next(iter(balanced_triplets(12, seed_offset=80_003)))
    records, key = build_records(
        triplet,
        episode_index=3,
        k=2,
        condition="cue_high",
    )
    assert set(records) == set(PROMPT_ARMS)
    assert {record["task_id"] for record in records.values()} == {
        key["task_id"]
    }
    for record in records.values():
        assert set(record) == {"task_id", "prompt", "prompt_sha256"}
        assert record["prompt_sha256"] == hashlib.sha256(
            record["prompt"].encode("utf-8")
        ).hexdigest()
        serialized = json.dumps(record)
        assert triplet["target"]["city_type"] not in serialized
        assert "target_matches" not in serialized
        assert "expected_poll" not in serialized


def test_contexts_cross_truth_and_hold_all_numbers_fixed():
    for episode_index, triplet in enumerate(
        balanced_triplets(120, seed_offset=80_000)
    ):
        numeric = []
        for condition in CONTEXT_CONDITIONS:
            _, key = build_records(
                triplet,
                episode_index=episode_index,
                k=2,
                condition=condition,
            )
            public = key["public"]
            numeric.append(
                (
                    public["reference_a"]["profile_index"],
                    public["reference_a"]["news"],
                    public["reference_a"]["ending_polls"],
                    public["reference_b"]["profile_index"],
                    public["reference_b"]["news"],
                    public["reference_b"]["ending_polls"],
                    public["target"]["profile_index"],
                    public["target"]["news"],
                    public["target"]["ending_polls"],
                    public["test_news"],
                )
            )
        assert all(value == numeric[0] for value in numeric)

    high = low = 0
    for triplet in balanced_triplets(120, seed_offset=80_000):
        if triplet["target"]["city_type"] == HIGH_TYPE:
            high += 1
        else:
            low += 1
    assert (high, low) == (60, 60)

    triplets = list(balanced_triplets(120, seed_offset=80_000))
    crossings = sum(
        (triplet["target"]["city_type"] == HIGH_TYPE
         and triplet["profile_index_target"] < 50.0)
        or (triplet["target"]["city_type"] != HIGH_TYPE
            and triplet["profile_index_target"] > 50.0)
        for triplet in triplets
    )
    assert crossings == 14
    assert len({triplet["profile_index_target"] for triplet in triplets}) > 80


def test_reference_geometry_has_matched_headroom_and_convergence():
    triplets = list(balanced_triplets(120, seed_offset=80_000))
    target = {
        k: _mae(triplets, compute_target_only, k) for k in PREFIX_LADDER
    }
    shrinkage = {
        k: _mae(triplets, compute_abc_shrinkage, k) for k in PREFIX_LADDER
    }
    assert target[5] < 0.75
    assert target[5] < target[1]
    assert target[1] - shrinkage[1] > 1.25
    assert target[2] - shrinkage[2] > 0.65
    assert all(shrinkage[k] < target[k] for k in PREFIX_LADDER)
    assert abs(shrinkage[5] - target[5]) < 0.1
    weights = [
        compute_abc_shrinkage(triplets[0], k)["reference_weight"]
        for k in PREFIX_LADDER
    ]
    assert all(
        current < previous
        for previous, current in zip(weights, weights[1:])
    )
    assert weights[0] > 0.70
    assert weights[-1] < 0.20
    assert [cases_for_prefix(k) for k in PREFIX_LADDER] == [1, 2, 4, 8, 16]


def test_wrong_context_cue_is_overridden_from_k1_to_k5():
    values = {1: [], 2: [], 5: []}
    for triplet in balanced_triplets(120, seed_offset=80_000):
        wrong = (
            "cue_low"
            if triplet["target"]["city_type"] == HIGH_TYPE
            else "cue_high"
        )
        truth_a = triplet["target_matches"] == "A"
        for k in values:
            result = compute_context_oracle(triplet, k, condition=wrong)
            p_a = result["p_match_a"]
            values[k].append(p_a if truth_a else 1.0 - p_a)
    means = {k: statistics.mean(rows) for k, rows in values.items()}
    assert means[2] > means[1]
    assert means[5] > means[2]
    assert means[5] > 0.9


def test_parser_and_terminal_response_semantics(tmp_path):
    parsed = _parse_reply(
        'preface {"rationale":"first","predicted_poll":20} '
        '{"rationale":"final","predicted_poll":57.25}'
    )
    assert parsed == {"rationale": "final", "predicted_poll": 57.25}
    assert _parse_reply('{"predicted_poll": 101}')["predicted_poll"] is None

    path = tmp_path / "responses.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "task_id": "received_unparseable",
                        "predicted_poll": None,
                        "response_received": True,
                    }
                ),
                json.dumps(
                    {
                        "task_id": "transport_failure",
                        "predicted_poll": None,
                        "response_received": False,
                    }
                ),
                json.dumps(
                    {
                        "task_id": "parsed",
                        "predicted_poll": 51.0,
                        "response_received": True,
                    }
                ),
            )
        )
        + "\n"
    )
    terminal, parsed_ids = _response_state(path)
    assert terminal == {"received_unparseable", "parsed"}
    assert parsed_ids == {"parsed"}
