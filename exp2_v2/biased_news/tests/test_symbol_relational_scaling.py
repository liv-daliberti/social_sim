"""The symbol relational/causal protocol must scale by sealed-test size only.

Every count in that pipeline used to be pinned to the 80-episode v1 task file.
These tests hold the derivation to two requirements: it reproduces the numbers the
v1 protocol froze, and it scales them without changing the protocol itself.
"""

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mechanistic_probe"))

from symbol_relational_common import (  # noqa: E402
    ARMS,
    CV_FOLDS,
    FACTORIAL_CELLS,
    RUN_REGISTRY,
    design_counts,
    run_spec,
)

PROBE_ROOT = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4" / "mechanistic_probe"
V1 = PROBE_ROOT / "qwen3_14b_symbol_relational_v1"
V2 = PROBE_ROOT / "qwen3_14b_symbol_relational_v2"


def manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "task_manifest.json").read_text())


@pytest.mark.skipif(not V1.exists(), reason="v1 relational run absent")
def test_derived_counts_reproduce_the_frozen_v1_protocol():
    """The v1 numbers were literals; they must survive as derivations."""
    counts = design_counts(manifest(V1))
    assert counts == {
        "episodes": 80,
        "development_episodes": 64,
        "sealed_test_episodes": 16,
        "all_records": 480,
        "label_extraction_records": 160,
        "patch_records": 320,
        "self_patch_pairs": 64,
        "min_patch_episodes": 14,
        "min_self_patch_pairs": 58,
    }


@pytest.mark.skipif(not V2.exists(), reason="v2 relational run absent")
def test_v2_scales_the_same_protocol():
    counts = design_counts(manifest(V2))
    assert counts["sealed_test_episodes"] == 88
    assert counts["development_episodes"] == 160
    # Structure is unchanged: the same records per episode, arm and condition.
    assert counts["label_extraction_records"] == counts["episodes"] * len(ARMS)
    assert counts["patch_records"] == counts["sealed_test_episodes"] * 2 * len(ARMS) * 5
    assert counts["self_patch_pairs"] == counts["sealed_test_episodes"] * 2 * len(ARMS)
    # Completeness floors keep the fractions the v1 protocol froze.
    v1 = design_counts(manifest(V1))
    assert counts["min_patch_episodes"] / counts["sealed_test_episodes"] == pytest.approx(
        v1["min_patch_episodes"] / v1["sealed_test_episodes"], abs=0.02
    )
    assert counts["min_self_patch_pairs"] / counts["self_patch_pairs"] == pytest.approx(
        v1["min_self_patch_pairs"] / v1["self_patch_pairs"], abs=0.02
    )


@pytest.mark.skipif(not V2.exists(), reason="v2 relational run absent")
def test_sealed_split_divides_evenly_across_design_cells():
    for run_dir in (V1, V2):
        counts = design_counts(manifest(run_dir))
        assert counts["development_episodes"] % FACTORIAL_CELLS == 0
        assert counts["sealed_test_episodes"] % FACTORIAL_CELLS == 0
        assert (counts["development_episodes"] // FACTORIAL_CELLS) % CV_FOLDS == 0


def test_registry_pins_each_run_to_its_own_task_file_and_source_probe():
    assert set(RUN_REGISTRY) == {
        "qwen3_14b_symbol_relational_v1",
        "qwen3_14b_symbol_relational_v2",
        "llama3_1_70b_symbol_relational_v1",
        "qwen3_32b_symbol_relational_v1",
        "qwen2_5_72b_symbol_relational_v1",
        "qwen3_14b_symbol_probe_v1",
        "qwen3_14b_symbol_probe_v2",
    }
    # A relational run and the probe run it draws from share a task file.
    for version in ("v1", "v2"):
        rel = RUN_REGISTRY[f"qwen3_14b_symbol_relational_{version}"]
        probe = RUN_REGISTRY[rel["source_probe_run"]]
        assert rel["tasks_sha256"] == probe["tasks_sha256"]
    v1, v2 = RUN_REGISTRY["qwen3_14b_symbol_relational_v1"], RUN_REGISTRY[
        "qwen3_14b_symbol_relational_v2"
    ]
    # The two runs must not share a task file, a source probe run, or a study name.
    for key in ("tasks_sha256", "task_manifest_sha256", "source_features_sha256",
                "source_results_sha256", "source_probe_run", "study"):
        assert v1[key] != v2[key], key
    with pytest.raises(ValueError, match="unregistered"):
        run_spec(Path("/tmp/qwen3_14b_symbol_relational_v3"))
    llama = RUN_REGISTRY["llama3_1_70b_symbol_relational_v1"]
    assert llama["replication_role"] == "cross_family_confirmation"
    assert llama["parent_study"] == "qwen3_14b_symbol_relational_v2"
    assert llama["transformer_layers"] == 80
    assert llama["tasks_sha256"] == v2["tasks_sha256"]
    qwen32 = RUN_REGISTRY["qwen3_32b_symbol_relational_v1"]
    assert qwen32["replication_role"] == "same_family_scale_confirmation"
    assert qwen32["source_receipt_required"] is True
    assert qwen32["transformer_layers"] == 64
    qwen72 = RUN_REGISTRY["qwen2_5_72b_symbol_relational_v1"]
    assert qwen72["replication_role"] == "cross_generation_scale_confirmation"
    assert qwen72["source_features_required"] is False
    assert qwen72["transformer_layers"] == 80
    for replication in (qwen32, qwen72):
        assert replication["parent_study"] == "qwen3_14b_symbol_relational_v2"
        assert replication["tasks_sha256"] == v2["tasks_sha256"]


@pytest.mark.skipif(not V2.exists(), reason="v2 relational run absent")
def test_frozen_v2_protocol_records_the_scaled_design():
    frozen = json.loads((V2 / "relational_protocol_manifest.json").read_text())
    design = frozen["task_design"]
    assert design["all_records"] == 1_488
    assert design["sealed_test_episodes"] == 88
    assert design["label_extraction_records"] == 496
    checks = frozen["stability_gate_before_cross_family_replication"]["checks"]
    assert any("at least 77 complete causal-patch episodes" in c for c in checks)
    assert any("at least 319 parsed self-patch pairs" in c for c in checks)
    # The protocol itself must be untouched: same arms, directions and conditions.
    causal = frozen["causal_patching"]
    assert causal["directions"] == ["correct_into_swapped", "swapped_into_correct"]
    assert causal["primary_evidence_depth"] == 0
    assert causal["primary_test"] == "exact one-sided episode sign-flip test"


PIPELINE = (
    "symbol_relational_common.py",
    "extract_symbol_label_states.py",
    "analyze_symbol_relational_probe.py",
    "patch_symbol_label_activations.py",
    "freeze_symbol_relational_protocol.py",
)
# Counts that were literals in the 80-episode version. Any of them reappearing in
# a comparison means a size was pinned again, which only fails on a GPU.
V1_PINNED_SIZES = (16, 64, 80, 160, 320, 480, 58, 14)


def test_no_pipeline_step_pins_a_v1_size_in_a_comparison():
    import re

    offenders = []
    pattern = re.compile(
        r"(?:!=|==)\s*\(?\s*(\d+)|len\([^)]*\)\s*(?:!=|==)\s*(\d+)"
    )
    for name in PIPELINE:
        for number, line in enumerate(
            (ROOT / "mechanistic_probe" / name).read_text().splitlines(), start=1
        ):
            if line.lstrip().startswith("#"):
                continue
            for match in pattern.finditer(line):
                value = int(match.group(1) or match.group(2))
                if value in V1_PINNED_SIZES:
                    offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, "sizes pinned to the 80-episode design:\n" + "\n".join(offenders)


def test_label_state_shape_expectation_is_derived_not_pinned():
    source = (ROOT / "mechanistic_probe" / "analyze_symbol_relational_probe.py").read_text()
    assert "(160, 41, 3, 2)" not in source
    assert "expected_shape = (len(rows), transformer_layers + 1, len(ANCHORS), 2)" in source
