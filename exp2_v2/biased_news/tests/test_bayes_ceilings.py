"""Checks for the Coin City Bayes ceiling and forecast-floor derivation.

The ceilings decide how a probe null is read, so the tests target the two ways
the derivation could be wrong: drift between the ceilings and the generator
constants they are supposed to describe, and a closed form that disagrees with
the simulated generator.
"""

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.compute_bayes_ceilings import (  # noqa: E402
    OUT_JSON,
    analytic_ceilings,
    monte_carlo,
)
from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    CASE_NOISE_SD,
    CITY_SLOPE_SD,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
)


@pytest.fixture(scope="module")
def simulated() -> dict:
    return monte_carlo(draws=60_000, seed=4242)


def test_residual_is_unidentifiable_without_target_rows():
    """At k=0 City C displays no rows, so no reader can score above zero."""
    assert analytic_ceilings()["residual_slope"]["k=0"] == 0.0


def test_residual_ceiling_stays_negligible_at_every_depth():
    """Four noisy rows cannot resolve a .03 spread; the ceiling must stay near zero."""
    ceilings = analytic_ceilings()["residual_slope"]
    assert max(ceilings.values()) < 0.02
    depths = [ceilings[f"k={k}"] for k in range(5)]
    assert depths == sorted(depths), "more target rows must not lower the ceiling"


def test_recoverable_targets_are_not_swept_up_in_the_null():
    """Regime and full slope stay recoverable; only the residual is not."""
    assert analytic_ceilings()["full_slope"]["k=0"] > 0.98


def test_closed_form_matches_the_simulated_generator(simulated):
    analytic = analytic_ceilings()
    for target in ("residual_slope", "full_slope"):
        for depth, value in analytic[target].items():
            assert simulated["r2_ceiling"][target][depth] == pytest.approx(value, abs=5e-3)


def test_reference_sampling_error_carries_no_city_c_information(simulated):
    """A reference city's residual is independent of City C's, by construction."""
    assert simulated["corr_reference_residual_with_city_c_residual"] == pytest.approx(0.0, abs=0.02)
    assert simulated["corr_regime_indicator_with_city_c_slope"] > 0.99


def test_arbitrary_label_regime_ceiling_is_below_one(simulated):
    """The two reference fits overlap, so the induced mapping is not certain."""
    assert 0.9 < simulated["regime_auc_ceiling_arbitrary_label_k0"] < 1.0
    assert simulated["regime_accuracy_ceiling_arbitrary_label_k0"] < 0.96


def test_artifact_records_the_generator_constants_it_describes():
    if not OUT_JSON.exists():
        pytest.skip("ceiling artifact has not been generated")
    stored = json.loads(OUT_JSON.read_text())["generator_constants"]
    assert stored["city_slope_sd"] == CITY_SLOPE_SD
    assert stored["case_noise_sd"] == CASE_NOISE_SD
    assert stored["strong_slope_mean"] == STRONG_SLOPE_MEAN
    assert stored["weak_slope_mean"] == WEAK_SLOPE_MEAN
