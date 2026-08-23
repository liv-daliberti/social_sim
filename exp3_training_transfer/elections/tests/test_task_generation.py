"""Regression tests for deterministic training-task generation."""

import random
import sys
from pathlib import Path

ELECTIONS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ELECTIONS_ROOT))

from scripts import generate_training_tasks as generator


def test_stochastic_generation_uses_seed(monkeypatch):
    state = {"E1": "type", "E2": "reliability", "E3": "tone", "E4": "volume"}
    monkeypatch.setattr(
        generator,
        "compute_exact_news_forecast",
        lambda: {"conditional": {"type|reliability|tone|volume": {"p_blue": 0.7}}},
    )

    seen_seeds = []

    def fake_simulate(*, seed=None, overrides=None):
        assert overrides is None
        seen_seeds.append(seed)
        return {"states": state}

    monkeypatch.setattr(generator, "simulate", fake_simulate)
    first = generator._generate_stochastic(3, seed=17)
    first_seeds = seen_seeds.copy()
    seen_seeds.clear()
    second = generator._generate_stochastic(3, seed=17)

    expected_rng = random.Random(17)
    expected_seeds = [expected_rng.randrange(2**63) for _ in range(3)]
    assert first == second
    assert first_seeds == seen_seeds == expected_seeds
