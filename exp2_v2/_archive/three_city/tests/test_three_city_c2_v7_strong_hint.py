import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.three_city_c2_v7 import (
    CONTEXT_CONDITIONS,
    PREFIX_LADDER,
    generate_triplet,
)
from eval.build_three_city_c2_v7_tasks import build_prompt
from eval.build_three_city_c2_v7_strong_hint_tasks import (
    STRONG_HINT_BLOCK,
    build_strong_prompt,
    strip_strong_hint,
    validate_strong_prompt,
)


def test_strong_hint_is_one_frozen_block_over_blind():
    triplet = generate_triplet(seed=70_000)
    for condition in CONTEXT_CONDITIONS:
        for k in PREFIX_LADDER:
            blind = build_prompt(
                triplet,
                k=k,
                condition=condition,
                arm="blind",
            )
            strong = build_strong_prompt(blind)
            assert strong.count(STRONG_HINT_BLOCK) == 1
            assert strip_strong_hint(strong) == blind
            validate_strong_prompt(strong)


def test_strong_hint_discloses_structure_but_not_answer():
    triplet = generate_triplet(seed=70_001)
    blind = build_prompt(
        triplet,
        k=2,
        condition="relevant",
        arm="blind",
    )
    strong = build_strong_prompt(blind)
    assert "one of two recurring response patterns" in strong
    assert "same response pattern as one of those cities" in strong
    assert (
        f"City C matches City {triplet['target_matches']}"
        not in STRONG_HINT_BLOCK
    )
    assert str(triplet["target"]["response"]) not in STRONG_HINT_BLOCK


def test_reasoning_and_output_contract_remain_common():
    triplet = generate_triplet(seed=70_002)
    blind = build_prompt(
        triplet,
        k=2,
        condition="none",
        arm="blind",
    )
    strong = build_strong_prompt(blind)
    common = (
        "vary from case to case",
        "do not provide step-by-step working",
        '"rationale": "one short sentence"',
        '"predicted_poll": <number from 0 to 100>',
    )
    for phrase in common:
        assert phrase in blind
        assert phrase in strong
