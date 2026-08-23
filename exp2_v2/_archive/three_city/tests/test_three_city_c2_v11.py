import statistics
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v11_confirmatory as analysis
from engine.three_city_c2_v11 import (
    PREFIX_LADDER,
    PROMPT_ARMS,
    balanced_triplets,
)
from eval import build_three_city_c2_v9_tasks as v9_builder
from eval.build_three_city_c2_v11_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    build_prompt,
    build_records,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)


def test_a_and_b_use_exact_v9_prompt_templates():
    triplet = next(iter(balanced_triplets(4)))
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
    triplet = next(iter(balanced_triplets(4)))
    prompts = {
        arm: build_prompt(triplet, k=2, arm=arm)
        for arm in PROMPT_ARMS
    }
    for arm, prompt in prompts.items():
        validate_arm_prompt(prompt, arm)
    assert strip_structural_clue(prompts["abc_structural_clue"]) == prompts["abc"]
    assert prompts["abc_structural_clue"].count(STRUCTURAL_CHOICE_CLUE) == 1
    assert len({target_suffix(prompt) for prompt in prompts.values()}) == 1


def test_continuous_design_is_balanced_and_identifiable():
    trips = list(balanced_triplets(120))
    cells = Counter()
    strengths = []
    agreements = []
    separations = []
    for triplet in trips:
        a = triplet["reference_a"]
        b = triplet["reference_b"]
        c = triplet["target"]
        pa, pb, pc = (
            float(a["profile_index"]),
            float(b["profile_index"]),
            float(c["profile_index"]),
        )
        profile_label = "A" if abs(pc - pa) < abs(pc - pb) else "B"
        response_label = (
            "A"
            if abs(float(c["response"]) - float(a["response"]))
            < abs(float(c["response"]) - float(b["response"]))
            else "B"
        )
        span = abs(pa - pb)
        strengths.append(2 * abs(pc - (pa + pb) / 2) / span)
        agreements.append(profile_label == response_label)
        separations.append(8 * abs(float(a["response"]) - float(b["response"])))
        cells[(triplet["high_reference"], profile_label)] += 1
    assert cells == Counter(
        {("A", "A"): 30, ("A", "B"): 30,
         ("B", "A"): 30, ("B", "B"): 30}
    )
    assert min(strengths) >= 0.54
    assert statistics.mean(agreements) >= 0.90
    assert statistics.median(separations) >= 5.0


def test_v11_analysis_adapter_uses_structural_arm():
    pairs = []
    for index in range(12):
        target = 50.0 + index
        matched = target + (-1.0 if index % 2 else 1.0)
        pair = {"episode_id": f"c2v11_{index:04d}"}
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
