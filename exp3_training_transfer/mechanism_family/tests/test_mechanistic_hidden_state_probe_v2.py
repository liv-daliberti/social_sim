from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


FAMILY = Path(__file__).resolve().parents[1]
PROBE = FAMILY / "mechanistic_probe"
sys.path.insert(0, str(PROBE))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROBE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load("exp3_probe_v2_tests", "analyze_probe_v2.py")
aux = load("exp3_probe_aux_tests", "build_auxiliary_targets.py")


def rows(worlds=("a", "a", "b", "b"), ks=(3, 3, 3, 3)):
    return [
        {"world": world, "k": k, "dev_fold": index % 2}
        for index, (world, k) in enumerate(zip(worlds, ks))
    ]


def test_development_group_residualization_uses_supplied_means() -> None:
    data = np.asarray((1.0, 3.0, 10.0, 14.0))
    sample = rows()
    means = v2.group_means(data, sample, pooled=False)
    np.testing.assert_allclose(
        v2.residualize(data, sample, means, pooled=False),
        np.asarray((-1.0, 1.0, -2.0, 2.0)),
    )
    test = [{"world": "a", "k": 3}, {"world": "b", "k": 3}]
    np.testing.assert_allclose(
        v2.residualize(np.asarray((4.0, 9.0)), test, means, pooled=False),
        np.asarray((2.0, -3.0)),
    )


def test_fair_r2_uses_development_baseline_not_test_mean() -> None:
    sample = rows(worlds=("a", "a", "a", "a"))
    target = np.asarray((1.0, 2.0, 3.0, 4.0))
    zero = np.zeros_like(target)
    metrics = v2.fair_metrics(zero, target, sample, pooled=False)
    assert metrics["macro_development_baseline_r2"] == 0.0
    assert metrics["macro_oracle_test_mean_r2"] < 0.0


def test_mean_alignment_removes_disclosure_translation() -> None:
    source = np.asarray(((0.0, 1.0), (2.0, 3.0), (10.0, 11.0), (12.0, 13.0)))
    shifted = source + np.asarray((100.0, -50.0))
    sample = rows()
    source_means = v2.feature_group_means(source, sample, pooled=False)
    cross_means = v2.feature_group_means(shifted, sample, pooled=False)
    aligned = v2.mean_align_features(
        shifted,
        sample,
        source_means,
        cross_means,
        pooled=False,
    )
    np.testing.assert_allclose(aligned, source)


def test_pooled_groups_include_evidence_depth() -> None:
    sample = rows(worlds=("a", "a", "a", "a"), ks=(3, 3, 6, 6))
    data = np.asarray((1.0, 3.0, 10.0, 14.0))
    means = v2.group_means(data, sample, pooled=True)
    assert set(means) == {("a", 3), ("a", 6)}
    np.testing.assert_allclose(
        v2.residualize(data, sample, means, pooled=True),
        np.asarray((-1.0, 1.0, -2.0, 2.0)),
    )


def test_auxiliary_slopes_are_explicit_and_deterministic() -> None:
    np.testing.assert_allclose(
        aux.slope(np.asarray((0.0, 1.0, 2.0)), np.asarray((1.0, 3.0, 5.0))),
        2.0,
    )
    assert aux.slope(np.ones(3), np.arange(3.0)) == 0.0
