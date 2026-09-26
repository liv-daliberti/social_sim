"""Sanity tests for the biased-news temporal DGP."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.temporal_dag import (
    generate_episode,
    generate_tasks,
    build_task,
    episode_stats,
    VARIANTS,
)


def test_episode_structure():
    ep = generate_episode(seed=0)
    T = ep["T"]
    assert 6 <= T <= 12
    assert ep["bias"] in ("biased", "neutral")
    assert len(ep["events"])    == T + 1
    assert len(ep["surveys"])   == T + 1
    assert len(ep["opinion_traj"]) == T + 2  # init + T+1 steps
    assert 0 <= ep["gold_survey"] <= 100
    assert 0.0 <= ep["gold_opinion"] <= 100.0


def test_episode_reproducible():
    ep1 = generate_episode(seed=42)
    ep2 = generate_episode(seed=42)
    assert ep1["bias"]         == ep2["bias"]
    assert ep1["surveys"]      == ep2["surveys"]
    assert ep1["opinion_traj"] == ep2["opinion_traj"]


def test_opinion_in_range():
    for seed in range(200):
        ep = generate_episode(seed=seed)
        for op in ep["opinion_traj"]:
            assert 0.0 <= op <= 100.0, f"opinion out of range: {op}"
        for sv in ep["surveys"]:
            assert 0 <= sv <= 100, f"survey out of range: {sv}"


def test_bias_dampens_effect():
    """Biased cities should show smaller opinion swings than neutral ones."""
    import statistics

    swings_biased  = []
    swings_neutral = []

    for seed in range(500):
        ep = generate_episode(seed=seed, T=10)
        traj = ep["opinion_traj"]
        swing = max(traj) - min(traj)
        if ep["bias"] == "biased":
            swings_biased.append(swing)
        else:
            swings_neutral.append(swing)

    mean_b = statistics.mean(swings_biased)
    mean_n = statistics.mean(swings_neutral)
    assert mean_b < mean_n, (
        f"Expected biased swing ({mean_b:.2f}) < neutral swing ({mean_n:.2f})"
    )


def test_build_task_variants():
    ep = generate_episode(seed=7)
    for v in ["V0", "V1", "V2"]:
        task = build_task(ep, variant=v, city_id=0)
        assert task["variant"] == v
        assert task["context_prefix"] == VARIANTS[v]
        assert len(task["history"]) == ep["T"]
        assert "settlement_value" in task
        assert "_hidden_bias" in task


def test_generate_tasks_count():
    tasks = generate_tasks(n_cities=10, variants=["V0", "V1"], seed=0)
    assert len(tasks) == 20  # 10 cities × 2 variants


def test_generate_tasks_same_city_different_variants():
    tasks = generate_tasks(n_cities=5, variants=["V0", "V1", "V2"], seed=0)
    # Tasks 0,1,2 should be same city with different variants
    t0, t1, t2 = tasks[0], tasks[1], tasks[2]
    assert t0["_hidden_bias"]        == t1["_hidden_bias"]        == t2["_hidden_bias"]
    assert t0["_hidden_opinion_init"] == t1["_hidden_opinion_init"] == t2["_hidden_opinion_init"]
    assert t0["variant"] == "V0"
    assert t1["variant"] == "V1"
    assert t2["variant"] == "V2"


def test_stats_smoke():
    stats = episode_stats(n=200, seed=0)
    assert "biased"  in stats
    assert "neutral" in stats
    assert stats["biased"]["final_std"]  > 0
    assert stats["neutral"]["final_std"] > 0
