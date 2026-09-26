"""Shared nonlinear mechanism grammar for the paired disclosed/undisclosed C3 study."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

SCENARIO_HORIZONS = (1, 3)
SCENARIO_SHOCKS = (-12.0, -4.0, 0.0, 4.0, 12.0)
SCENARIO_LABELS = tuple(chr(ord("A") + i) for i in range(
    len(SCENARIO_HORIZONS) * len(SCENARIO_SHOCKS)
))
GAIN_GRID = np.linspace(0.10, 1.30, 49)
INPUT_POOL = np.asarray((-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0))


@dataclass(frozen=True)
class World:
    name: str
    split: str
    block: str
    law: str
    path: str
    phi: float
    mediator_phi: float
    coupling: float
    feedback: float
    saturation: float
    threshold: float
    asymmetry: float
    distractor_coef: float
    gain_lo: float
    gain_hi: float
    noise: float
    baseline: float
    clip: tuple[float, float]
    drivers: int
    domain: str
    driver_labels: tuple[str, ...]
    outcome_label: str
    period: str

    @property
    def mechanism_key(self) -> tuple[str, str, int]:
        """Canonical graph/function identity; surfaces and numeric parameters are excluded."""
        return (self.law, self.path, self.drivers)

    def on_surface_of(self, other: "World") -> "World":
        """Put a candidate mechanism on another task's native measurement surface."""
        labels = other.driver_labels[: self.drivers]
        if len(labels) < self.drivers:
            labels = labels + tuple(f"driver_{i+1}" for i in range(len(labels), self.drivers))
        return replace(
            self,
            baseline=other.baseline,
            clip=other.clip,
            domain=other.domain,
            driver_labels=labels,
            outcome_label=other.outcome_label,
            period=other.period,
        )


def _world(
    name: str,
    *,
    split: str,
    block: str,
    law: str,
    path: str,
    domain: str,
    driver: str,
    outcome: str,
    period: str,
    phi: float = 0.0,
    mediator_phi: float = 0.0,
    coupling: float = 0.0,
    feedback: float = 0.0,
    saturation: float = 7.0,
    threshold: float = 4.0,
    asymmetry: float = 1.25,
    distractor_coef: float = 0.0,
    gain_lo: float = 0.20,
    gain_hi: float = 0.90,
    noise: float = 1.5,
    baseline: float = 50.0,
    clip: tuple[float, float] = (0.0, 100.0),
    drivers: int = 1,
    driver_2: str = "background pressure",
) -> World:
    return World(
        name=name,
        split=split,
        block=block,
        law=law,
        path=path,
        phi=phi,
        mediator_phi=mediator_phi,
        coupling=coupling,
        feedback=feedback,
        saturation=saturation,
        threshold=threshold,
        asymmetry=asymmetry,
        distractor_coef=distractor_coef,
        gain_lo=gain_lo,
        gain_hi=gain_hi,
        noise=noise,
        baseline=baseline,
        clip=clip,
        drivers=drivers,
        domain=domain,
        driver_labels=(driver,) if drivers == 1 else (driver, driver_2),
        outcome_label=outcome,
        period=period,
    )


CATALOG = (
    _world("direct_linear", split="train", block="primitive", law="linear", path="direct",
           domain="Northport", driver="net dispatch", outcome="backlog", period="shift",
           baseline=48.0, noise=1.2),
    _world("persistent_linear", split="train", block="primitive", law="linear", path="persistent",
           phi=0.62, domain="Lake Ward", driver="staffing pressure", outcome="wait index",
           period="day", baseline=55.0, noise=1.6),
    _world("mediated_linear", split="train", block="primitive", law="linear", path="mediated",
           phi=0.22, mediator_phi=0.48, coupling=0.78, domain="Redwood Basin",
           driver="rain anomaly", outcome="runoff index", period="week", baseline=44.0, noise=1.4),
    _world("feedback_linear", split="train", block="primitive", law="linear", path="feedback",
           phi=0.38, mediator_phi=0.34, feedback=0.36, domain="Cobalt Exchange",
           driver="order imbalance", outcome="liquidity index", period="session",
           baseline=51.0, noise=1.8),
    _world("two_driver_linear", split="train", block="primitive", law="linear", path="persistent",
           phi=0.42, drivers=2, distractor_coef=0.22, domain="Mesa Grid",
           driver="demand shock", driver_2="temperature load", outcome="reserve index",
           period="hour", baseline=58.0, noise=1.5),
    _world("direct_saturating", split="train", block="primitive", law="saturating", path="direct",
           saturation=6.0, domain="Harbor Market", driver="arrival surge", outcome="clearance index",
           period="tide", baseline=47.0, noise=1.3),
    _world("direct_threshold", split="train", block="primitive", law="threshold", path="direct",
           threshold=4.0, asymmetry=1.30, domain="Pine Clinic", driver="case surge",
           outcome="load index", period="day", baseline=52.0, noise=1.7),
    _world("persistent_saturating", split="train", block="primitive", law="saturating",
           path="persistent", phi=0.48, saturation=7.0, domain="Delta Stores",
           driver="delivery shock", outcome="stock index", period="cycle", baseline=46.0, noise=2.0),
    _world("mediated_feedback_linear", split="test", block="topology_composition", law="linear",
           path="mediated_feedback", phi=0.24, mediator_phi=0.38, coupling=0.68, feedback=0.18,
           domain="Aurora Transit", driver="service shock", outcome="delay index", period="interval",
           baseline=54.0, noise=1.8),
    _world("mediated_saturating", split="test", block="nonlinear_composition", law="saturating",
           path="mediated", phi=0.20, mediator_phi=0.44, coupling=0.82, saturation=6.5,
           domain="Marsh Ecology", driver="nutrient pulse", outcome="bloom index", period="fortnight",
           baseline=43.0, noise=1.6),
    _world("persistent_threshold", split="test", block="mixed_composition", law="threshold",
           path="persistent", phi=0.58, threshold=3.5, asymmetry=1.20, domain="Summit Supply",
           driver="shipment shock", outcome="shortage index", period="week", baseline=57.0, noise=1.9),
    _world("feedback_saturating_extrap", split="test", block="parameter_extrapolation",
           law="saturating", path="feedback", phi=0.44, mediator_phi=0.40, feedback=0.32,
           saturation=8.0, gain_lo=1.02, gain_hi=1.28, noise=4.2, domain="Estuary Control",
           driver="gate adjustment", outcome="salinity index", period="cycle", baseline=49.0),
)

BY_NAME = {world.name: world for world in CATALOG}
TRAIN = tuple(world for world in CATALOG if world.split == "train")
TEST = tuple(world for world in CATALOG if world.split == "test")


def _effect(world: World, x: float, gain: float) -> float:
    if world.law == "linear":
        return gain * x
    if world.law == "saturating":
        return gain * world.saturation * np.tanh(x / world.saturation)
    if world.law == "threshold":
        if x >= 0:
            return gain * max(0.0, x - world.threshold)
        return gain * world.asymmetry * min(0.0, x + world.threshold)
    raise ValueError(f"unknown law: {world.law}")


def transition(world: World, state: tuple[float, float], inputs: Sequence[float], gain: float):
    """Advance native-scale latent deviations (outcome state, mediator state)."""
    z_prev, mediator_prev = state
    primary = _effect(world, float(inputs[0]), gain)
    distractor = world.distractor_coef * float(inputs[1]) if world.drivers > 1 else 0.0
    drive = primary + distractor
    if world.path == "direct":
        z = drive
        mediator = 0.0
    elif world.path == "persistent":
        z = world.phi * z_prev + drive
        mediator = 0.0
    elif world.path == "mediated":
        mediator = world.mediator_phi * mediator_prev + drive
        z = world.phi * z_prev + world.coupling * mediator_prev
    elif world.path == "feedback":
        mediator = world.mediator_phi * mediator_prev + drive + world.feedback * z_prev
        z = world.phi * z_prev + drive + world.feedback * mediator_prev
    elif world.path == "mediated_feedback":
        mediator = world.mediator_phi * mediator_prev + drive + world.feedback * z_prev
        z = world.phi * z_prev + world.coupling * mediator_prev
    else:
        raise ValueError(f"unknown path: {world.path}")
    return float(z), float(mediator)


def balanced_inputs(world: World, length: int, rng: np.random.Generator) -> np.ndarray:
    cols = []
    for _ in range(world.drivers):
        values = np.resize(INPUT_POOL, length).copy()
        rng.shuffle(values)
        cols.append(values)
    return np.stack(cols, axis=1)


def expected_series(world: World, inputs: np.ndarray, gain: float):
    state = (0.0, 0.0)
    means = []
    for row in np.asarray(inputs, dtype=float):
        state = transition(world, state, row, gain)
        means.append(float(np.clip(world.baseline + state[0], *world.clip)))
    return np.asarray(means), state


def simulate_series(world: World, inputs: np.ndarray, gain: float, rng: np.random.Generator):
    means, state = expected_series(world, inputs, gain)
    observed = np.clip(means + rng.normal(0.0, world.noise, len(means)), *world.clip)
    return observed, means, state


def scenario_grid() -> list[dict]:
    scenarios = []
    for horizon in SCENARIO_HORIZONS:
        for shock in SCENARIO_SHOCKS:
            scenarios.append({
                "label": SCENARIO_LABELS[len(scenarios)],
                "horizon": int(horizon),
                "shock": float(shock),
            })
    return scenarios


def forecast_scenarios(world: World, history_inputs: np.ndarray, gain: float) -> np.ndarray:
    _, initial = expected_series(world, history_inputs, gain)
    answers = []
    for scenario in scenario_grid():
        state = initial
        for step in range(1, int(scenario["horizon"]) + 1):
            future = np.zeros(world.drivers)
            if step == 1:
                future[0] = float(scenario["shock"])
            state = transition(world, state, future, gain)
        answers.append(float(np.clip(world.baseline + state[0], *world.clip)))
    return np.asarray(answers)


def response_pairs() -> list[tuple[int, int]]:
    pairs = []
    width = len(SCENARIO_SHOCKS)
    zero_offset = SCENARIO_SHOCKS.index(0.0)
    for horizon_i in range(len(SCENARIO_HORIZONS)):
        zero_i = horizon_i * width + zero_offset
        for shock_i, shock in enumerate(SCENARIO_SHOCKS):
            if shock != 0.0:
                pairs.append((horizon_i * width + shock_i, zero_i))
    return pairs


def response_vector(targets: Sequence[float]) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    return np.asarray([values[i] - values[j] for i, j in response_pairs()])


def _global_prior_response() -> np.ndarray:
    responses = []
    zero_history = np.zeros((0, 1))
    for world in TRAIN:
        history = np.zeros((0, world.drivers)) if world.drivers > 1 else zero_history
        for gain in np.linspace(world.gain_lo, world.gain_hi, 9):
            targets = forecast_scenarios(world, history, float(gain))
            responses.append(response_vector(targets))
    return np.mean(np.stack(responses), axis=0)


GLOBAL_PRIOR_RESPONSE = _global_prior_response()


def population_prior_targets(world: World, last_observed: float) -> np.ndarray:
    """Best frozen catalog-level response policy; it does not inspect mechanism or episode."""
    values = np.full(len(SCENARIO_LABELS), float(last_observed))
    for response, (i, j) in zip(GLOBAL_PRIOR_RESPONSE, response_pairs()):
        values[i] = values[j] + response
    return np.clip(values, *world.clip)


def mechanism_description(world: World) -> str:
    law = {
        "linear": "The immediate driver effect is g times the signed driver value.",
        "saturating": (
            f"The driver effect is g*{world.saturation:g}*tanh(driver/{world.saturation:g}), "
            "so large shocks saturate."
        ),
        "threshold": (
            f"Driver magnitudes up to {world.threshold:g} have no effect; beyond that threshold "
            f"the excess is multiplied by g, with negative effects multiplied by {world.asymmetry:g}."
        ),
    }[world.law]
    path = {
        "direct": "Each period replaces the latent outcome deviation with the current driver effect.",
        "persistent": f"The latent outcome retains {world.phi:g} of its previous deviation, then adds the current effect.",
        "mediated": (
            f"The driver enters a mediator retaining {world.mediator_phi:g}; the outcome retains "
            f"{world.phi:g} and receives {world.coupling:g} times the mediator from the previous period."
        ),
        "feedback": (
            f"Driver and outcome interact through a feedback state (outcome retention {world.phi:g}, "
            f"feedback-state retention {world.mediator_phi:g}, coupling {world.feedback:g})."
        ),
        "mediated_feedback": (
            f"The driver enters a mediator retaining {world.mediator_phi:g}; the mediator receives "
            f"{world.feedback:g} feedback from the prior outcome, and the outcome receives "
            f"{world.coupling:g} of the prior mediator while retaining {world.phi:g}."
        ),
    }[world.path]
    distractor = ""
    if world.drivers > 1:
        distractor = f" The second driver has a fixed coefficient {world.distractor_coef:g}."
    return (
        f"The stable mechanism is known. {law} {path}{distractor} The episode-specific g is fixed "
        f"within a trajectory and lies between {world.gain_lo:g} and {world.gain_hi:g}. Observations "
        f"have Gaussian noise with standard deviation {world.noise:g}."
    )


def log_likelihood(world: World, inputs: np.ndarray, observed: np.ndarray, gain: float) -> float:
    expected, _ = expected_series(world, inputs, gain)
    residual = np.asarray(observed) - expected
    variance = world.noise ** 2
    return float(-0.5 * np.sum(residual ** 2 / variance + np.log(2.0 * np.pi * variance)))


def _logmeanexp(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    peak = float(array.max())
    return peak + float(np.log(np.mean(np.exp(array - peak))))


def oracle_forecast(
    surface: World,
    calibrations: Sequence[dict],
    target_inputs: np.ndarray,
    target_observed: np.ndarray,
    *,
    disclosed: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Grid posterior over structure and target gain; calibration gains are marginalized."""
    candidates = (surface,) if disclosed else tuple(
        candidate.on_surface_of(surface) for candidate in CATALOG if candidate.drivers == surface.drivers
    )
    weighted = []
    diagnostics: dict[str, float] = {}
    for candidate in candidates:
        structure_ll = 0.0
        for calibration in calibrations:
            structure_ll += _logmeanexp(
                log_likelihood(candidate, calibration["inputs"], calibration["observed"], gain)
                for gain in GAIN_GRID
            )
        for gain in GAIN_GRID:
            ll = structure_ll + log_likelihood(candidate, target_inputs, target_observed, float(gain))
            forecast = forecast_scenarios(candidate, target_inputs, float(gain))
            weighted.append((ll, forecast, candidate.name, float(gain)))
    log_weights = np.asarray([item[0] for item in weighted])
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    prediction = np.sum(
        np.stack([item[1] for item in weighted]) * weights[:, None], axis=0
    )
    for weight, item in zip(weights, weighted):
        diagnostics[item[2]] = diagnostics.get(item[2], 0.0) + float(weight)
    return prediction, diagnostics


def validate_catalog() -> None:
    assert len(TRAIN) == 8 and len(TEST) == 4
    assert len(BY_NAME) == len(CATALOG)
    train_keys = {world.mechanism_key for world in TRAIN}
    test_keys = {world.mechanism_key for world in TEST}
    assert train_keys.isdisjoint(test_keys), (train_keys & test_keys)
    assert max(world.gain_hi for world in TRAIN) < max(world.gain_hi for world in TEST)
    assert max(world.noise for world in TRAIN) < max(world.noise for world in TEST)
    for world in CATALOG:
        assert world.gain_lo < world.gain_hi
        assert world.clip[0] < world.baseline < world.clip[1]
        rng = np.random.default_rng(7)
        inputs = balanced_inputs(world, 20, rng)
        expected, _ = expected_series(world, inputs, (world.gain_lo + world.gain_hi) / 2)
        assert np.all(np.isfinite(expected))
        assert np.mean((expected <= world.clip[0]) | (expected >= world.clip[1])) < 0.10


validate_catalog()
