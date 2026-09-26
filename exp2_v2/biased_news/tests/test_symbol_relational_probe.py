from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "mechanistic_probe"
sys.path.insert(0, str(PROBE))

import analyze_symbol_relational_probe as analysis
import patch_symbol_label_activations as patching
import symbol_relational_common as common


def test_label_subset_and_reference_roles_are_balanced() -> None:
    tasks, _ = common.load_tasks(common.SOURCE_RUN_DIR)
    rows = common.label_tasks(tasks)
    assert len(rows) == 160
    assert Counter((row["split"], row["arm"]) for row in rows) == {
        ("dev", "abc_context"): 64,
        ("dev", "abc_wrong_context"): 64,
        ("test", "abc_context"): 16,
        ("test", "abc_wrong_context"): 16,
    }
    role_counts = Counter()
    for row in rows:
        matching, nonmatching = common.reference_roles(row)
        assert {matching, nonmatching} == {"city_a", "city_b"}
        role_counts[(row["split"], row["arm"], matching)] += 1
    assert set(role_counts.values()) == {8, 32}


def test_patch_window_is_fixed_width_and_clamped() -> None:
    assert analysis.three_layer_window(1) == [1, 2, 3]
    assert analysis.three_layer_window(2) == [1, 2, 3]
    assert analysis.three_layer_window(20) == [19, 20, 21]
    assert analysis.three_layer_window(39) == [37, 38, 39]
    assert analysis.three_layer_window(79, last_effective_layer=79) == [77, 78, 79]
    assert analysis.three_layer_window(79, last_effective_layer=79) == [77, 78, 79]
    with pytest.raises(ValueError, match="downstream transformer block"):
        analysis.three_layer_window(40)


def test_exact_sign_flip_is_episode_level_and_one_sided() -> None:
    result = analysis.exact_sign_flip(np.asarray([1.0, 1.0]))
    assert result["scheme"] == "exact episode-level sign flip"
    assert result["n"] == 2
    assert result["mean"] == 1.0
    assert result["one_sided_p"] == 0.25


def test_correct_swapped_pair_may_differ_only_at_two_label_tokens() -> None:
    patching.validate_matched_pair(
        correct_positions=[2, 3],
        wrong_positions=[2, 3],
        correct_ids=[10, 11, 20, 21, 12],
        wrong_ids=[10, 11, 30, 31, 12],
    )


def test_primary_representation_and_gate_are_frozen() -> None:
    assert common.PRIMARY_REPRESENTATION == "city_c_minus_matching_reference"
    assert common.PRIMARY_TARGET == "regime"
    assert common.PERMUTATION_REPEATS == 10_000
    assert common.REPRESENTATIONS == (
        "city_c",
        "matching_reference",
        "city_c_minus_matching_reference",
        "city_c_minus_nonmatching_reference",
        "matching_minus_nonmatching_reference",
    )
