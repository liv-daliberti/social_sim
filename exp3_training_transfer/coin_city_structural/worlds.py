"""Locked worlds for the Coin City A-to-B structural-transfer experiment.

Training contains only Structure A, the direct Exp2 relationship.  Evaluation
crosses that familiar structure with a persistent mediated Structure B and
crosses the familiar Coin City surface with a held-out Coin Harbor surface.
The equations and the internal structure labels are never rendered to models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


STRUCTURES = ("direct_a", "mediated_b")
HORIZONS = (1, 3)
SHOCK_INDEX = (-2, -1, 0, 1, 2)
SCENARIO_LABELS = tuple(chr(ord("A") + i) for i in range(10))


@dataclass(frozen=True)
class Domain:
    name: str
    label: str
    driver: str
    outcome: str
    period: str
    baseline: float
    clip: tuple[float, float]
    input_pool: tuple[float, ...]
    shock_unit: float
    noise: float
    direct_reference_context: str
    mediated_reference_context: str
    direct_target_context: str
    mediated_target_context: str

    @property
    def scenarios(self) -> tuple[dict, ...]:
        rows = []
        for horizon in HORIZONS:
            for index in SHOCK_INDEX:
                rows.append({
                    "label": SCENARIO_LABELS[len(rows)],
                    "horizon": horizon,
                    "shock": float(index * self.shock_unit),
                })
        return tuple(rows)


COIN_CITY = Domain(
    name="coin_city",
    label="Coin City",
    driver="net campaign news",
    outcome="candidate support",
    period="week",
    baseline=50.0,
    clip=(0.0, 100.0),
    input_pool=(-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0),
    shock_unit=6.0,
    noise=1.6,
    direct_reference_context=(
        "Residents receive campaign developments through synchronized citywide alerts. "
        "Reactions occur mostly during the same week as the news."
    ),
    mediated_reference_context=(
        "Campaign developments usually spread through neighborhood conversations and "
        "local organizations. Reactions may build gradually and persist after the "
        "original story has faded."
    ),
    direct_target_context=(
        "Residents receive campaign developments through synchronized citywide alerts. "
        "Reactions occur mostly during the same week as the news."
    ),
    mediated_target_context=(
        "Campaign developments usually spread through neighborhood conversations and "
        "local organizations. Reactions may build gradually and persist after the "
        "original story has faded."
    ),
)


COIN_HARBOR = Domain(
    name="coin_harbor",
    label="Coin Harbor",
    driver="net shipping bulletin",
    outcome="cargo-flow index",
    period="tide cycle",
    baseline=180.0,
    clip=(40.0, 320.0),
    input_pool=(-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0),
    shock_unit=12.0,
    noise=4.5,
    direct_reference_context=(
        "Shipping bulletins reach vessels through synchronized fleetwide radio alerts. "
        "Responses occur mostly during the same tide cycle as the bulletin."
    ),
    mediated_reference_context=(
        "Shipping bulletins usually spread through dockside crews and harbor groups. "
        "Responses may build gradually and persist after the original bulletin has faded."
    ),
    direct_target_context=(
        "Shipping bulletins reach vessels through synchronized fleetwide radio alerts. "
        "Responses occur mostly during the same tide cycle as the bulletin."
    ),
    mediated_target_context=(
        "Shipping bulletins usually spread through dockside crews and harbor groups. "
        "Responses may build gradually and persist after the original bulletin has faded."
    ),
)


DOMAINS = {domain.name: domain for domain in (COIN_CITY, COIN_HARBOR)}


def sample_parameters(structure: str, rng: np.random.Generator, *, gain_band=None) -> dict:
    if gain_band is None:
        gain_band = (0.22, 0.92)
    gain = float(rng.uniform(*gain_band))
    if structure == "direct_a":
        return {"gain": gain, "rho": 0.0, "phi": 0.0, "lambda": 1.0}
    if structure == "mediated_b":
        return {
            "gain": gain,
            "rho": float(rng.uniform(0.48, 0.74)),
            "phi": float(rng.uniform(0.18, 0.42)),
            "lambda": float(rng.uniform(0.68, 0.94)),
        }
    raise ValueError(f"unknown structure: {structure}")


def balanced_inputs(domain: Domain, length: int, rng: np.random.Generator) -> np.ndarray:
    values = np.resize(np.asarray(domain.input_pool, dtype=float), length).copy()
    rng.shuffle(values)
    return values


def transition(structure: str, state: tuple[float, float], shock: float,
               parameters: dict) -> tuple[float, float]:
    """Advance outcome deviation and mediator on their native display scale."""
    previous, mediator_previous = state
    drive = float(parameters["gain"]) * float(shock)
    if structure == "direct_a":
        return drive, 0.0
    if structure == "mediated_b":
        mediator = float(parameters["rho"]) * mediator_previous + drive
        outcome = (float(parameters["phi"]) * previous
                   + float(parameters["lambda"]) * mediator)
        return outcome, mediator
    raise ValueError(f"unknown structure: {structure}")


def expected_series(domain: Domain, structure: str, inputs: Sequence[float],
                    parameters: dict) -> tuple[np.ndarray, tuple[float, float]]:
    state = (0.0, 0.0)
    means = []
    for shock in inputs:
        state = transition(structure, state, float(shock), parameters)
        means.append(float(np.clip(domain.baseline + state[0], *domain.clip)))
    return np.asarray(means, dtype=float), state


def simulate_series(domain: Domain, structure: str, inputs: Sequence[float],
                    parameters: dict, rng: np.random.Generator) -> dict:
    means, _ = expected_series(domain, structure, inputs, parameters)
    observed = np.clip(
        means + rng.normal(0.0, domain.noise, len(means)), *domain.clip
    )
    return {
        "inputs": np.asarray(inputs, dtype=float).round(4).tolist(),
        "observed": observed.round(4).tolist(),
        "expected": means.round(6).tolist(),
    }


def forecast_scenarios(domain: Domain, structure: str, history_inputs: Sequence[float],
                       parameters: dict) -> np.ndarray:
    _, initial = expected_series(domain, structure, history_inputs, parameters)
    answers = []
    for scenario in domain.scenarios:
        state = initial
        for step in range(1, int(scenario["horizon"]) + 1):
            shock = float(scenario["shock"]) if step == 1 else 0.0
            state = transition(structure, state, shock, parameters)
        answers.append(float(np.clip(domain.baseline + state[0], *domain.clip)))
    return np.asarray(answers, dtype=float)


def response_pairs() -> tuple[tuple[int, int], ...]:
    pairs = []
    width = len(SHOCK_INDEX)
    zero = SHOCK_INDEX.index(0)
    for horizon_index in range(len(HORIZONS)):
        zero_index = horizon_index * width + zero
        for shock_index, shock in enumerate(SHOCK_INDEX):
            if shock:
                pairs.append((horizon_index * width + shock_index, zero_index))
    return tuple(pairs)


def response_vector(targets: Sequence[float]) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    return np.asarray([values[left] - values[right]
                       for left, right in response_pairs()], dtype=float)


def context_for(domain: Domain, structure: str, *, target: bool = False) -> str:
    suffix = "target_context" if target else "reference_context"
    stem = "direct" if structure == "direct_a" else "mediated"
    return str(getattr(domain, f"{stem}_{suffix}"))
