import math
import json
import sys
from pathlib import Path

import matplotlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_three_city_c2_v7_confirmatory as analysis
from engine.three_city_c2_v7 import PREFIX_LADDER
from eval.build_three_city_c2_v7_tasks import build_records
from engine.three_city_c2_v7 import balanced_triplets

matplotlib.use("Agg")


def _synthetic_rows_and_keys():
    rows = []
    keys = {}
    models = tuple(analysis._DISPLAY)
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
            hierarchical = key["baselines"]["empirical_hierarchical"][
                "predicted_poll"
            ]
            gold = key["gold"]["expected_poll"]
            for model in models:
                for arm, prediction in (
                    ("blind", target),
                    ("hint", hierarchical),
                ):
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "task_id": key["task_id"],
                            "episode_id": key["episode_id"],
                            "condition": "relevant",
                            "k": k,
                            "prediction": prediction,
                            "gold": gold,
                            "absolute_error": abs(prediction - gold),
                            "naive": key["baselines"]["naive"][
                                "predicted_poll"
                            ],
                            "target_only": target,
                            "hierarchical": hierarchical,
                            "pooled_references": key["baselines"][
                                "pooled_references"
                            ]["predicted_poll"],
                            "rationale": "synthetic pipeline check",
                        }
                    )
    return rows, keys, models


def test_structure_use_coefficient_has_intended_anchors():
    rows, _, models = _synthetic_rows_and_keys()
    paired = analysis._paired_rows(
        rows,
        model=models[0],
        condition="relevant",
        k=2,
    )
    assert analysis._beta([pair["blind"] for pair in paired]) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert analysis._beta([pair["hint"] for pair in paired]) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_bootstrap_and_paper_figure_pipeline(tmp_path, monkeypatch):
    rows, keys, models = _synthetic_rows_and_keys()
    monkeypatch.setattr(analysis, "_BOOTSTRAP", 100)
    paired = analysis._paired_rows(
        rows,
        model=models[0],
        condition="relevant",
        k=2,
    )
    result = analysis._bootstrap_primary(paired, seed=20260728)
    assert result["n"] == 12
    assert result["hint_beta"] == pytest.approx(1.0, abs=1e-12)
    assert result["blind_beta"] == pytest.approx(0.0, abs=1e-12)
    assert all(math.isfinite(value) for value in result["beta_improvement_ci"])
    assert all(math.isfinite(value) for value in result["blind_beta_ci"])
    assert all(math.isfinite(value) for value in result["hint_beta_ci"])

    secondary = analysis._secondary_results(rows, models)
    assert secondary[models[0]]["relevant"]["2"]["n"] == 12
    sensitivity = analysis._context_weight_sensitivity(keys)
    assert set(sensitivity["2"]) == {"0", "0.5", "1", "2"}

    figure_path = tmp_path / "three_city_c2_v7_main.png"
    analysis._make_figure(
        figure_path,
        rows=rows,
        keys=keys,
        models=models,
    )
    assert figure_path.stat().st_size > 50_000
    assert figure_path.with_suffix(".pdf").stat().st_size > 10_000


def test_latest_response_prefers_received_and_parsed_records(tmp_path):
    path = tmp_path / "responses_test.jsonl"
    records = [
        {
            "model": "gpt-5.4",
            "arm": "blind",
            "task_id": "task",
            "predicted_poll": None,
            "response_received": False,
        },
        {
            "model": "gpt-5.4",
            "arm": "blind",
            "task_id": "task",
            "predicted_poll": None,
            "response_received": True,
        },
        {
            "model": "gpt-5.4",
            "arm": "blind",
            "task_id": "task",
            "predicted_poll": 57.2,
            "response_received": True,
        },
        {
            "model": "gpt-5.4",
            "arm": "blind",
            "task_id": "task",
            "predicted_poll": None,
            "response_received": False,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    latest = analysis._latest_responses([path])
    assert latest[("gpt-5.4", "blind", "task")]["predicted_poll"] == 57.2


def test_full_analysis_cli_pipeline(tmp_path, monkeypatch):
    _, keys, models = _synthetic_rows_and_keys()
    answer_key = tmp_path / "answer_key.jsonl"
    answer_key.write_text(
        "".join(
            json.dumps(key) + "\n"
            for key in keys.values()
        )
    )
    responses_dir = tmp_path / "responses"
    responses_dir.mkdir()
    for model in models:
        records = []
        for key in keys.values():
            for arm, method in (
                ("blind", "target_only"),
                ("hint", "empirical_hierarchical"),
            ):
                records.append(
                    {
                        "model": model,
                        "arm": arm,
                        "task_id": key["task_id"],
                        "predicted_poll": key["baselines"][method][
                            "predicted_poll"
                        ],
                        "rationale": "synthetic pipeline check",
                        "response_received": True,
                    }
                )
        (responses_dir / f"responses_{model}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

    outdir = tmp_path / "analysis"
    monkeypatch.setattr(analysis, "_BOOTSTRAP", 20)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_three_city_c2_v7_confirmatory.py",
            "--answer-key",
            str(answer_key),
            "--responses-dir",
            str(responses_dir),
            "--outdir",
            str(outdir),
        ],
    )
    analysis.main()

    result = json.loads(
        (outdir / "confirmatory_results.json").read_text()
    )
    assert set(result["primary"]) == set(models)
    assert "secondary_by_condition_and_k" in result
    assert "hierarchical_context_weight_sensitivity" in result
    assert (outdir / "three_city_c2_v7_main.png").stat().st_size > 50_000
    assert (outdir / "three_city_c2_v7_main.pdf").stat().st_size > 10_000
    assert "Preregistered interpretation rule" in (
        outdir / "confirmatory_summary.md"
    ).read_text()
