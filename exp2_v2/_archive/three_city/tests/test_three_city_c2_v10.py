import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v10_confirmatory as analysis
from engine.three_city_c2_v10 import PREFIX_LADDER, PROMPT_ARMS, balanced_triplets
from eval import build_three_city_c2_v9_tasks as v9_builder
from eval.build_three_city_c2_v10_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    build_prompt,
    build_records,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)


def test_a_and_b_are_exact_v9_prompts():
    triplet = next(iter(balanced_triplets(2)))
    for k in PREFIX_LADDER:
        for arm in ("c_only", "abc"):
            assert build_prompt(triplet, k=k, arm=arm) == v9_builder.build_prompt(
                triplet,
                k=k,
                arm=arm,
            )


def test_clue_reveals_structure_but_not_city_choice():
    lower = STRUCTURAL_CHOICE_CLUE.lower()
    assert "one of cities a and b" in lower
    assert "which reference" in lower
    assert "closer to city a" not in lower
    assert "closer to city b" not in lower
    assert "city a is" not in lower
    assert "city b is" not in lower
    assert "distance" not in lower


def test_structural_arm_differs_only_by_fixed_clue():
    triplet = next(iter(balanced_triplets(2)))
    prompts = {
        arm: build_prompt(triplet, k=2, arm=arm)
        for arm in PROMPT_ARMS
    }
    for arm, prompt in prompts.items():
        validate_arm_prompt(prompt, arm)
    assert strip_structural_clue(prompts["abc_structural_clue"]) == prompts["abc"]
    assert prompts["abc_structural_clue"].count(STRUCTURAL_CHOICE_CLUE) == 1
    assert len({target_suffix(prompt) for prompt in prompts.values()}) == 1


def test_numerical_records_match_v9_except_ids():
    triplet = next(iter(balanced_triplets(2)))
    _, v10 = build_records(triplet, episode_index=0, k=2)
    _, v9 = v9_builder.build_records(triplet, episode_index=0, k=2)
    for key in set(v10) - {"task_id", "episode_id"}:
        assert v10[key] == v9[key]


def test_v10_analysis_adapter_uses_structural_arm():
    pairs = []
    for index in range(12):
        target = 50.0 + index
        matched = target + (-1.0 if index % 2 else 1.0)
        pair = {"episode_id": f"c2v10_{index:04d}"}
        for arm, prediction in (
            ("c_only", target),
            ("abc", matched),
            ("abc_structural_clue", matched),
        ):
            pair[arm] = {
                "prediction": prediction,
                "absolute_error": abs(prediction - 55.0),
                "target_only": target,
                "abc_shrinkage": matched,
            }
        pairs.append(pair)
    result = analysis._statistics(pairs, draws=100, seed=7)
    assert set(result["arms"]) == set(PROMPT_ARMS)
    assert "structural_clue_minus_abc" in result["contrasts"]
    assert result["contrasts"]["structural_clue_minus_abc"][
        "mae_improvement"
    ] == pytest.approx(0.0)
