import math
import sys
from pathlib import Path

import matplotlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v9_confirmatory as analysis
from engine.three_city_c2_v9 import (
    CASES_BY_PREFIX,
    PREFIX_LADDER,
    PROMPT_ARMS,
    balanced_triplets,
    cases_for_prefix,
    compute_abc_shrinkage,
    compute_target_only,
    gold_expected_poll,
)
from eval.build_three_city_c2_v9_tasks import (
    RELEVANCE_HINT_BLOCK,
    build_prompt,
    build_records,
    strip_relevance_hint,
    target_suffix,
    validate_arm_prompt,
)

matplotlib.use("Agg")


def test_five_round_case_ladder():
    assert PREFIX_LADDER == (1, 2, 3, 4, 5)
    assert CASES_BY_PREFIX == {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
    assert [cases_for_prefix(k) for k in PREFIX_LADDER] == [1, 2, 4, 8, 16]


def test_continuous_profiles_are_nonextreme_bracketed_and_balanced():
    triplets = list(balanced_triplets(120))
    high_labels = []
    scores = []
    responses = []
    for triplet in triplets:
        a = float(triplet["reference_a"]["profile_index"])
        b = float(triplet["reference_b"]["profile_index"])
        c = float(triplet["target"]["profile_index"])
        assert 0.0 < min(a, b, c) < max(a, b, c) < 100.0
        assert min(a, b) < c < max(a, b)
        high_labels.append("A" if a > b else "B")
        scores.extend((a, b, c))
        responses.append(float(triplet["target"]["response"]))
    assert high_labels.count("A") == high_labels.count("B") == 60
    assert len(set(scores)) > 250
    assert len(set(responses)) == 120


def test_information_arms_and_guidance_are_cleanly_separated():
    triplet = next(iter(balanced_triplets(2)))
    prompts = {
        arm: build_prompt(triplet, k=2, arm=arm)
        for arm in PROMPT_ARMS
    }
    for arm, prompt in prompts.items():
        validate_arm_prompt(prompt, arm)
    assert "CITY A\n" not in prompts["c_only"]
    assert "CITY B\n" not in prompts["c_only"]
    assert "CITY A\n" in prompts["abc"]
    assert "CITY B\n" in prompts["abc"]
    assert RELEVANCE_HINT_BLOCK not in prompts["abc"]
    assert prompts["abc"] == strip_relevance_hint(prompts["abc_relevance"])
    assert len({target_suffix(prompt) for prompt in prompts.values()}) == 1


def test_frozen_baselines_have_large_start_and_matched_convergence():
    triplets = list(balanced_triplets(120))
    target_mae = {}
    abc_mae = {}
    weights = {}
    for k in PREFIX_LADDER:
        target_errors = []
        abc_errors = []
        round_weights = []
        for triplet in triplets:
            gold = gold_expected_poll(triplet)
            target = compute_target_only(triplet, k)
            abc = compute_abc_shrinkage(triplet, k)
            target_errors.append(abs(float(target["predicted_poll"]) - gold))
            abc_errors.append(abs(float(abc["predicted_poll"]) - gold))
            round_weights.append(float(abc["reference_weight"]))
        target_mae[k] = sum(target_errors) / len(target_errors)
        abc_mae[k] = sum(abc_errors) / len(abc_errors)
        weights[k] = sum(round_weights) / len(round_weights)
    assert target_mae[1] - abc_mae[1] > 1.50
    assert target_mae[2] - abc_mae[2] > 0.75
    assert all(abc_mae[k] < target_mae[k] for k in (1, 2, 3, 4))
    assert abs(target_mae[5] - abc_mae[5]) < 0.10
    assert target_mae[5] < 0.90
    assert abc_mae[5] < 0.90
    assert weights[1] > 0.70
    assert weights[5] < 0.25
    assert all(weights[a] > weights[b] for a, b in zip(PREFIX_LADDER, PREFIX_LADDER[1:]))


def _synthetic_rows_and_keys():
    rows = []
    keys = {}
    models = tuple(analysis._DISPLAY)
    for episode_index, triplet in enumerate(balanced_triplets(12, seed_offset=92_000)):
        for k in PREFIX_LADDER:
            _, key = build_records(triplet, episode_index=episode_index, k=k)
            keys[key["task_id"]] = key
            target = float(key["baselines"]["target_only"]["predicted_poll"])
            abc = float(key["baselines"]["abc_shrinkage"]["predicted_poll"])
            predictions = {"c_only": target, "abc": abc, "abc_relevance": abc}
            gold = float(key["gold"]["expected_poll"])
            for model in models:
                for arm in PROMPT_ARMS:
                    prediction = predictions[arm]
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "task_id": key["task_id"],
                            "episode_id": key["episode_id"],
                            "k": k,
                            "prediction": prediction,
                            "gold": gold,
                            "absolute_error": abs(prediction - gold),
                            "target_only": target,
                            "abc_shrinkage": abc,
                        }
                    )
    return rows, keys, models


def test_analysis_pairing_slopes_and_figure(tmp_path):
    rows, keys, models = _synthetic_rows_and_keys()
    paired = analysis._paired(rows, model=models[0], k=2)
    assert len(paired) == 12
    stats = analysis._statistics(paired, draws=100, seed=7)
    assert stats["arms"]["c_only"]["pooling_slope"] == pytest.approx(0.0)
    assert stats["arms"]["abc"]["pooling_slope"] == pytest.approx(1.0)
    assert stats["arms"]["abc_relevance"]["pooling_slope"] == pytest.approx(1.0)
    curves = {}
    for model_index, model in enumerate(models):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = analysis._statistics(
                analysis._paired(rows, model=model, k=k),
                draws=30,
                seed=1000 * model_index + k,
            )
    path = tmp_path / "structure.png"
    analysis._make_structure_figure(
        path,
        curves=curves,
        keys=keys,
        models=models,
        interim=False,
    )
    assert path.stat().st_size > 10_000
    assert path.with_suffix(".pdf").stat().st_size > 10_000
