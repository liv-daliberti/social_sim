from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "mechanistic_probe"
sys.path.insert(0, str(PROBE))

import analyze_behavior_screen as behavior_analysis
import analyze_probe as analysis
import build_probe_tasks as builder
import launch_model_extension as model_extension
import screen_qwen_behavior as behavior_screen


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_toy_selection_is_cell_balanced_and_episode_disjoint() -> None:
    episodes = read_jsonl(builder.DESIGN / "episodes.jsonl")
    selected, assignment = builder.select_episodes(
        episodes,
        episodes_per_cell=16,
        test_per_cell=4,
    )

    counts = {}
    for row in selected:
        cell = (row["strong_reference_city"], bool(row["target_strong"]))
        split, fold = assignment[int(row["episode"])]
        counts[(cell, split)] = counts.get((cell, split), 0) + 1
        if split == "test":
            assert fold is None
        else:
            assert fold in range(4)

    for cell in builder.CELL_ORDER:
        assert counts[(cell, "dev")] == 12
        assert counts[(cell, "test")] == 4
    assert len(assignment) == 64
    assert len(set(assignment)) == 64


def test_matched_probe_arms_preserve_numeric_content() -> None:
    episodes = read_jsonl(builder.DESIGN / "episodes.jsonl")
    prompts, _ = builder.prompt_maps()
    selected, assignment = builder.select_episodes(
        episodes,
        episodes_per_cell=16,
        test_per_cell=4,
    )
    records = builder.make_records(selected, assignment, prompts)

    assert len(records) == 64 * 2 * 3
    by_condition = {}
    for row in records:
        key = (row["episode"], row["c_cases"])
        by_condition.setdefault(key, []).append(row)
    for rows in by_condition.values():
        assert {row["arm"] for row in rows} == set(builder.ARMS)
        numeric = {
            tuple(builder.NUMBER.findall(row["prompt"]))
            for row in rows
        }
        assert len(numeric) == 1


def test_dual_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(48, 20))
    test = rng.normal(size=(20, 20))
    weights = rng.normal(size=20)
    train_target = train @ weights
    test_target = test @ weights

    design = analysis.prepare_dual(train, test)
    prediction = analysis.dual_predict(design, train_target, alpha=1e-4)
    metrics = analysis.regression_metrics(prediction, test_target)
    assert metrics["r2"] > 0.99
    assert metrics["pearson"] > 0.99


def test_auc_and_cross_validation_are_leakage_free() -> None:
    rng = np.random.default_rng(9)
    labels = np.tile(np.asarray((-1.0, 1.0)), 24)
    features = rng.normal(size=(48, 16))
    features[:, 0] = labels + rng.normal(scale=0.05, size=48)
    folds = np.tile(np.arange(4), 12)

    alpha, score, all_scores = analysis.cross_validated_alpha(
        train=features,
        target=labels,
        folds=folds,
        target_name="regime",
        alphas=analysis.ALPHAS,
    )
    assert alpha in analysis.ALPHAS
    assert set(all_scores) == {f"{value:g}" for value in analysis.ALPHAS}
    assert score > 0.99
    assert analysis.roc_auc(labels, labels > 0) == 1.0


def test_permutation_null_uses_held_out_labels() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(48, 8))
    test = rng.normal(size=(16, 8))
    weights = rng.normal(size=8)
    train_target = train @ weights
    test_target = test @ weights

    null = analysis.permutation_null(
        train=train,
        test=test,
        train_target=train_target,
        test_target=test_target,
        alpha=1e-4,
        target_name="slope",
        repeats=1_000,
    )
    assert null["scheme"].startswith("held-out target-label permutation")
    assert null["observed_score"] > 0.99
    assert null["one_sided_p"] < 0.01


def test_within_regime_target_removes_regime_mean() -> None:
    rows = [
        {
            "target_strong": True,
            "target_slope": 0.92,
            "residual_slope": 0.02,
        },
        {
            "target_strong": False,
            "target_slope": 0.23,
            "residual_slope": -0.02,
        },
    ]
    np.testing.assert_allclose(
        analysis.target_values(rows, "residual_slope"),
        np.asarray((0.02, -0.02)),
    )


def test_behavior_screen_reuses_sealed_test_and_baselines() -> None:
    tasks, manifest = behavior_screen.load_tasks(behavior_screen.DEFAULT_TASKS_DIR)
    test_ids = [row["sample_id"] for row in tasks if row["split"] == "test"]
    assert len(test_ids) == 96
    assert len(set(test_ids)) == 96

    reference = behavior_screen.DEFAULT_TASKS_DIR / "behavior_test.jsonl"
    assert behavior_screen.verify_existing(reference, test_ids)
    validated = behavior_analysis.validate_screen(
        behavior_screen.DEFAULT_TASKS_DIR,
        set(test_ids),
        manifest["tasks_sha256"],
    )
    assert validated["model_label"] == "Qwen3-8B"

    baseline = behavior_analysis.baselines(tasks)
    assert np.isclose(baseline["unit_slope"]["mae"], 2.686081172093712)
    assert np.isclose(baseline["regime_mean"]["mae"], 0.1466844318073739)

    task_by_id = {row["sample_id"]: row for row in tasks}
    paired = behavior_analysis.paired_behavior_contrasts(reference, task_by_id)
    context_shift = paired["by_prefix"]["k0"]["context_minus_no_context"][
        "regime_aligned_slope"
    ]
    assert context_shift["n"] == 16
    assert np.isclose(context_shift["mean"], 0.078125)
    assert paired["bootstrap_repeats"] == 10_000

def test_cross_architecture_extension_is_pinned_and_dry_by_default() -> None:
    assert set(model_extension.MODELS) == {"qwen3_4b", "llama3_1_8b"}
    for model_key, item in model_extension.MODELS.items():
        assert len(item["model_commit"]) == 40
        command = model_extension.command(model_key)
        assert command[0:2] == ["sbatch", "--parsable"]
        assert f"MODEL={item['model']}" in command[2]
        assert f"MODEL_COMMIT={item['model_commit']}" in command[2]
        assert command[-1].endswith("run_open_model_probe.sbatch")

