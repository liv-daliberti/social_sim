from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


FAMILY = Path(__file__).resolve().parents[1]
PROBE = FAMILY / "mechanistic_probe"
sys.path.insert(0, str(PROBE))

import launch_probe as launcher



def load_probe_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROBE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_probe_module("exp3_hidden_state_analysis", "analyze_probe.py")
builder = load_probe_module("exp3_hidden_state_builder", "build_probe_tasks.py")

def read_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in (PROBE / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_frozen_tasks_are_paired_balanced_and_episode_disjoint() -> None:
    tasks = read_tasks()
    summary = builder.validate_records(tasks)
    assert summary == {
        "record_count": 1_440,
        "episode_count": 240,
        "development_episodes": 180,
        "test_episodes": 60,
    }
    episode_splits = {}
    for row in tasks:
        episode_splits.setdefault(row["episode_id"], set()).add(row["split"])
    assert all(len(value) == 1 for value in episode_splits.values())

    counts = Counter(
        (row["world"], row["split"], row["disclosure"], row["k"])
        for row in tasks
    )
    for world in builder.EXPECTED_WORLDS:
        for disclosure in builder.DISCLOSURES:
            for k in builder.K_VALUES:
                assert counts[(world, "dev", disclosure, k)] == 45
                assert counts[(world, "test", disclosure, k)] == 15


def test_disclosure_pairs_preserve_numeric_targets() -> None:
    tasks = read_tasks()
    pairs = {}
    for row in tasks:
        pairs.setdefault(row["task_id"], []).append(row)
    assert len(pairs) == 720
    for rows in pairs.values():
        assert {row["disclosure"] for row in rows} == {"disclosed", "undisclosed"}
        assert len({row["numeric_hash"] for row in rows}) == 1
        assert len({row["target_gain"] for row in rows}) == 1
        assert len({tuple(row["truth_response"]) for row in rows}) == 1
        assert len({tuple(row["prior_response"]) for row in rows}) == 1


def test_targets_remove_registered_between_world_shortcuts() -> None:
    gain_rows = [
        {"world": "mediated_feedback_linear", "target_gain": 0.20},
        {"world": "mediated_feedback_linear", "target_gain": 0.90},
        {"world": "feedback_saturating_extrap", "target_gain": 1.02},
        {"world": "feedback_saturating_extrap", "target_gain": 1.28},
    ]
    np.testing.assert_allclose(
        analysis.target_values(gain_rows, "gain"),
        np.asarray((-0.5, 0.5, -0.5, 0.5)),
    )
    response_rows = [
        {
            "truth_response": [5.0] * 8,
            "prior_response": [1.0] * 8,
        },
        {
            "truth_response": [-3.0] * 8,
            "prior_response": [1.0] * 8,
        },
    ]
    np.testing.assert_allclose(
        analysis.target_values(response_rows, "response_residual"),
        np.asarray(([1.0] * 8, [-1.0] * 8)),
    )


def test_dual_ridge_supports_multioutput_targets() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(80, 12))
    test = rng.normal(size=(30, 12))
    weights = rng.normal(size=(12, 8))
    train_target = train @ weights
    test_target = test @ weights
    prediction = analysis.dual_predict(
        analysis.prepare_dual(train, test), train_target, alpha=1e-4
    )
    assert analysis.r2_score(prediction, test_target) > 0.99


def test_permutation_stays_within_world() -> None:
    rows = [{"world": "a"}] * 4 + [{"world": "b"}] * 4
    target = np.arange(8, dtype=float)
    permuted = analysis.permute_within_world(
        target, rows, np.random.default_rng(3)
    )
    assert set(permuted[:4]) == set(target[:4])
    assert set(permuted[4:]) == set(target[4:])


def test_launcher_resolves_registered_pilot_without_submitting() -> None:
    launcher.preflight()
    model = launcher.MODELS["qwen3_8b"]
    assert launcher.model_snapshot(model["model"], model["commit"]).is_dir()
    for endpoint, job_id in model["jobs"].items():
        adapter = launcher.resolve_adapter("qwen3_8b", endpoint, job_id)
        assert (adapter / "adapter_model.safetensors").is_file()
        command = launcher.submission_command("qwen3_8b", endpoint, adapter)
        assert command[0:2] == ["sbatch", "--parsable"]
        assert "ADAPTER=" in command[2]
    assert "ADAPTER=" not in launcher.submission_command("qwen3_8b", "base", None)[2]
