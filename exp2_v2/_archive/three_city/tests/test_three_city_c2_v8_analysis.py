import math
import sys
from pathlib import Path

import matplotlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v8_confirmatory as analysis
from engine.three_city_c2_v8 import (
    CONTEXT_CONDITIONS,
    PREFIX_LADDER,
    PROMPT_ARMS,
    balanced_triplets,
)
from eval.build_three_city_c2_v8_tasks import build_records

matplotlib.use("Agg")


def _synthetic_rows_and_keys():
    rows = []
    keys = {}
    models = tuple(analysis._DISPLAY)
    deltas = {
        "blind": 0.3,
        "hint": 0.9,
        "strong_hint": 1.4,
    }
    for episode_index, triplet in enumerate(
        balanced_triplets(12, seed_offset=81_000)
    ):
        for k in PREFIX_LADDER:
            no_context_prediction = {}
            for condition in CONTEXT_CONDITIONS:
                _, key = build_records(
                    triplet,
                    episode_index=episode_index,
                    k=k,
                    condition=condition,
                )
                keys[key["task_id"]] = key
                target = key["baselines"]["target_only"]["predicted_poll"]
                mixture = key["baselines"]["abc_shrinkage"]["predicted_poll"]
                no_context_prediction = {
                    "blind": target,
                    "hint": 0.5 * (target + mixture),
                    "strong_hint": mixture,
                }
                attenuation = 1.0 - 0.9 * ((k - 1) / 4.0)
                for model in models:
                    for arm in PROMPT_ARMS:
                        prediction = no_context_prediction[arm]
                        if condition == "cue_high":
                            prediction += deltas[arm] * attenuation
                        elif condition == "cue_low":
                            prediction -= deltas[arm] * attenuation
                        gold = key["gold"]["expected_poll"]
                        rows.append(
                            {
                                "model": model,
                                "arm": arm,
                                "task_id": key["task_id"],
                                "episode_id": key["episode_id"],
                                "condition": condition,
                                "k": k,
                                "target_type": key["gold"]["target_type"],
                                "prediction": prediction,
                                "gold": gold,
                                "absolute_error": abs(prediction - gold),
                                "target_only": target,
                                "abc_shrinkage": mixture,
                                "privileged_structure": key["baselines"][
                                    "privileged_structure_ceiling"
                                ]["predicted_poll"],
                                "privileged_context": key["baselines"][
                                    "privileged_context_oracle"
                                ]["predicted_poll"],
                                "rationale": "synthetic pipeline check",
                            }
                        )
    return rows, keys, models


def test_structure_use_coefficient_has_zero_half_one_anchors():
    rows, _, models = _synthetic_rows_and_keys()
    paired = analysis._paired_structure(rows, model=models[0], k=2)
    assert len(paired) == 12
    expected = {"blind": 0.0, "hint": 0.5, "strong_hint": 1.0}
    for arm, value in expected.items():
        assert analysis._beta(
            [pair[arm] for pair in paired]
        ) == pytest.approx(value, abs=1e-12)


def test_complete_cell_pairing_drops_an_episode_symmetrically():
    rows, _, models = _synthetic_rows_and_keys()
    model = models[0]
    complete = analysis._paired_structure(rows, model=model, k=2)
    missing = [
        row
        for row in rows
        if not (
            row["model"] == model
            and row["episode_id"] == "c2v8_0000"
            and row["condition"] == "none"
            and row["k"] == 2
            and row["arm"] == "strong_hint"
        )
    ]
    paired = analysis._paired_structure(missing, model=model, k=2)
    assert len(complete) == 12
    assert len(paired) == 11
    assert "c2v8_0000" not in {row["episode_id"] for row in paired}


def test_context_metrics_capture_selectivity_direction_and_override():
    rows, _, models = _synthetic_rows_and_keys()
    model = models[0]
    paired0 = analysis._paired_context(rows, model=model, k=1)
    stats0 = analysis._context_statistics(paired0, draws=100, seed=7)
    for arm, delta in (
        ("blind", 0.3),
        ("hint", 0.9),
        ("strong_hint", 1.4),
    ):
        cell = stats0["arms"][arm]
        assert cell["relevant_movement"] == pytest.approx(delta)
        assert cell["orthogonal_movement"] == pytest.approx(0.0)
        assert cell["cue_direction"] == pytest.approx(1.0)
        assert cell["misleading_displacement"] == pytest.approx(delta)

    override_pairs = analysis._paired_context_override(rows, model=model)
    override = analysis._override_statistics(
        override_pairs,
        draws=100,
        seed=8,
    )
    assert override["n"] == 12
    for arm, delta in (
        ("blind", 0.3),
        ("hint", 0.9),
        ("strong_hint", 1.4),
    ):
        assert override["arms"][arm]["override_reduction"] == pytest.approx(
            0.9 * delta
        )


def test_statistics_and_both_figure_pipelines(tmp_path):
    rows, keys, models = _synthetic_rows_and_keys()
    curves = {}
    context = {}
    for model_index, model in enumerate(models):
        curves[model] = {}
        context[model] = {}
        for k_index, k in enumerate(PREFIX_LADDER):
            curves[model][str(k)] = analysis._structure_statistics(
                analysis._paired_structure(rows, model=model, k=k),
                draws=30,
                seed=1000 * model_index + k_index,
                include_beta=False,
            )
            context[model][str(k)] = analysis._context_statistics(
                analysis._paired_context(rows, model=model, k=k),
                draws=30,
                seed=10_000 + 1000 * model_index + k_index,
            )

    primary = analysis._structure_statistics(
        analysis._paired_structure(rows, model=models[0], k=2),
        draws=100,
        seed=9,
        include_beta=True,
    )
    assert primary["n"] == 12
    assert primary["arms"]["hint"]["beta"] == pytest.approx(0.5)
    assert all(
        math.isfinite(value)
        for value in primary["arms"]["strong_hint"]["beta_ci"]
    )

    structure = tmp_path / "structure.png"
    context_path = tmp_path / "context.png"
    analysis._make_structure_figure(
        structure,
        curves=curves,
        keys=keys,
        models=models,
        interim=False,
    )
    analysis._make_context_figure(
        context_path,
        context=context,
        oracle=analysis._oracle_context_curves(keys),
        models=models,
        interim=False,
    )
    for path in (
        structure,
        structure.with_suffix(".pdf"),
        context_path,
        context_path.with_suffix(".pdf"),
    ):
        assert path.stat().st_size > 10_000
