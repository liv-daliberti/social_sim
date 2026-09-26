import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v7_confirmatory as frozen
from analysis import analyze_three_city_c2_v7_strong_hint as strong_analysis
from analysis import plot_three_city_c2_v7_paper as paper
from engine.three_city_c2_v7 import PREFIX_LADDER, balanced_triplets
from eval.build_three_city_c2_v7_tasks import build_records

matplotlib.use("Agg")


def _three_arm_fixture():
    rows = []
    keys = {}
    models = tuple(frozen._DISPLAY)
    for episode_index, triplet in enumerate(balanced_triplets(12)):
        for k in PREFIX_LADDER:
            _, key = build_records(
                triplet,
                episode_index=episode_index,
                k=k,
                condition="relevant",
            )
            keys[key["task_id"]] = key
            target = key["baselines"]["target_only"]["predicted_poll"]
            structured = key["baselines"]["empirical_hierarchical"][
                "predicted_poll"
            ]
            predictions = {
                "blind": target,
                "hint": 0.5 * (target + structured),
                "strong_hint": structured,
            }
            for model in models:
                for arm, prediction in predictions.items():
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "task_id": key["task_id"],
                            "episode_id": key["episode_id"],
                            "condition": "relevant",
                            "k": k,
                            "prediction": prediction,
                            "gold": key["gold"]["expected_poll"],
                            "absolute_error": abs(
                                prediction - key["gold"]["expected_poll"]
                            ),
                            "naive": key["baselines"]["naive"][
                                "predicted_poll"
                            ],
                            "target_only": target,
                            "hierarchical": structured,
                            "pooled_references": key["baselines"][
                                "pooled_references"
                            ]["predicted_poll"],
                            "rationale": "synthetic three-arm check",
                        }
                    )
    return rows, keys, models


def test_three_arm_paper_figure_uses_common_episode_pairs(
    tmp_path,
    monkeypatch,
):
    rows, keys, models = _three_arm_fixture()
    monkeypatch.setattr(frozen, "_BOOTSTRAP", 50)

    paired = paper._paired_arm_rows(
        rows,
        model=models[0],
        condition="relevant",
        k=2,
        arms=("blind", "hint", "strong_hint"),
    )
    assert len(paired) == 12
    assert set(paired[0]) == {
        "episode_id",
        "blind",
        "hint",
        "strong_hint",
    }

    figure_path = tmp_path / "three_arm.png"
    counts, beta_counts = paper._make_figure(
        figure_path,
        rows=rows,
        keys=keys,
        models=models,
        arms=("blind", "hint", "strong_hint"),
        interim=False,
    )
    assert figure_path.stat().st_size > 50_000
    assert figure_path.with_suffix(".pdf").stat().st_size > 10_000
    assert beta_counts == {model: 12 for model in models}
    for arm in ("blind", "hint", "strong_hint"):
        for model in models:
            assert counts[arm][model] == [12, 12, 12, 12]


def test_common_pairing_excludes_episode_missing_from_one_arm():
    rows, _, models = _three_arm_fixture()
    missing = next(
        row
        for row in rows
        if row["model"] == models[0]
        and row["arm"] == "strong_hint"
        and row["k"] == 2
    )
    rows.remove(missing)
    paired = paper._paired_arm_rows(
        rows,
        model=models[0],
        condition="relevant",
        k=2,
        arms=("blind", "hint", "strong_hint"),
    )
    assert len(paired) == 11


def test_three_arm_bootstrap_recovers_structure_use_anchors():
    rows, _, models = _three_arm_fixture()
    paired = paper._paired_arm_rows(
        rows,
        model=models[0],
        condition="relevant",
        k=2,
        arms=("blind", "hint", "strong_hint"),
    )
    result = strong_analysis._bootstrap_three_arm(
        paired,
        seed=20260728,
        draws=100,
    )
    assert result["n"] == 12
    assert abs(result["arms"]["blind"]["beta"]) < 1e-12
    assert abs(result["arms"]["hint"]["beta"] - 0.5) < 1e-12
    assert abs(result["arms"]["strong_hint"]["beta"] - 1.0) < 1e-12
    assert (
        abs(result["contrasts"]["strong_over_hint"]["beta_increase"] - 0.5)
        < 1e-12
    )
