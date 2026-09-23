from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper import generate_coin_city_mechanistic_comparison as generator


def test_six_model_mechanistic_roster_and_claims() -> None:
    expected_labels = [
        "Qwen3-4B",
        "Qwen3-8B",
        "Llama-3.1-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Qwen2.5-72B",
    ]
    assert [spec.label for spec in generator.RUNS] == expected_labels

    runs = [generator.load_run(spec) for spec in generator.RUNS]
    residual_correct = []
    residual_p = []
    residual_all_arms = []
    no_context_full_slope = []
    inverted_regime_true = []
    inverted_regime_cue = []

    for loaded in runs:
        conditions = loaded["results"]["primary"]["conditions"]
        for k in (0, 4):
            residual = conditions[f"k{k}:residual_slope"]
            residual_p.append(
                residual["permutation_null_on_correct_context_test"]["one_sided_p"]
            )
            residual_correct.append(
                residual["selected_test"]["abc_context"]["probe"]["true_target"]["r2"]
            )
            residual_all_arms.extend(
                residual["selected_test"][arm]["probe"]["true_target"]["r2"]
                for arm in ("abc_context", "abc_no_context", "abc_wrong_context")
            )

            slope = conditions[f"k{k}:slope"]["selected_test"]
            no_context_full_slope.append(
                slope["abc_no_context"]["probe"]["true_target"]["r2"]
            )

            regime = conditions[f"k{k}:regime"]["selected_test"]["abc_wrong_context"][
                "probe"
            ]
            inverted_regime_true.append(regime["true_target"]["roc_auc"])
            inverted_regime_cue.append(regime["cue_target"]["roc_auc"])

    assert len(residual_correct) == 12
    assert round(min(residual_correct), 3) == -0.531
    assert round(max(residual_correct), 3) == 0.003
    assert round(min(residual_p), 3) == 0.221
    assert round(max(residual_p), 3) == 0.927
    assert all(value > 0.05 for value in residual_p)
    assert sum(value <= 0 for value in residual_all_arms) == 33
    assert all(value < 0 for value in no_context_full_slope)
    assert round(min(inverted_regime_true), 3) == 0.000
    assert round(max(inverted_regime_true), 3) == 0.109
    assert round(min(inverted_regime_cue), 3) == 0.891
    assert round(max(inverted_regime_cue), 3) == 1.000

    qwen_input_full_slope = []
    input_residual = []
    for loaded in runs:
        control = loaded["results"]["pooling_controls"][
            "mean_pool_source_layer_0"
        ]["conditions"]
        regime_values = [
            control[f"k{k}:regime"]["selected_test"]["abc_context"]["probe"][
                "true_target"
            ]["roc_auc"]
            for k in (0, 4)
        ]
        assert regime_values == [1, 1]
        slope_values = [
            control[f"k{k}:slope"]["selected_test"]["abc_context"]["probe"][
                "true_target"
            ]["r2"]
            for k in (0, 4)
        ]
        input_residual.extend(
            control[f"k{k}:residual_slope"]["selected_test"]["abc_context"]["probe"][
                "true_target"
            ]["r2"]
            for k in (0, 4)
        )
        if loaded["spec"].label == "Llama-3.1-8B":
            assert [round(value, 3) for value in slope_values] == [0.756, 0.827]
        else:
            qwen_input_full_slope.extend(slope_values)

    assert round(min(qwen_input_full_slope), 3) == 0.988
    assert round(max(qwen_input_full_slope), 3) == 0.993
    assert all(value < 0 for value in input_residual)

    summary = generator.summary_table(runs)
    for row in (
        "Qwen3-4B & 1.000/1.000 & .960/.885 & -.389/-.227 & .927/.614",
        "Llama-3.1-8B & 1.000/.984 & .913/.731 & -.126/-.207 & .380/.572",
        "Qwen3-32B & 1.000/1.000 & .967/.811 & -.269/.003 & .477/.221",
    ):
        assert row in summary

    appendix = " ".join(
        (
            generator.ROOT / "paper/appendix.tex"
        ).read_text().split()
    )
    for claim in (
        "Six-model correct-context probe summary",
        "all 12 model--$k$ cells",
        "33 of 36 arm-level residual scores are nonpositive",
        "$p=.221$--$.927$",
        "not specific to the Qwen checkpoints",
    ):
        assert claim in appendix
