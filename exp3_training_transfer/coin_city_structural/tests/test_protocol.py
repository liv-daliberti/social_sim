import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worlds import COIN_CITY, COIN_HARBOR, forecast_scenarios, response_vector  # noqa: E402
from make_paper_outputs import (  # noqa: E402
    APPENDIX_MODELS,
    EXPECTED_GROUPS,
    MODELS,
    SEEDS,
    endpoint_path,
    paired_override_interval,
    registered_estimates,
    render_cues,
    render_diagnostics,
    render_primary,
    render_summary,
    transfer_bootstrap_seed,
)


def test_endpoint_path_supports_training_and_base_layouts(tmp_path):
    training = tmp_path / "training"
    nested = training / "debug_123" / "eval_results" / "301.scores.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}\n")
    stochastic = training / "stochastic_n5.scores.jsonl"
    stochastic.write_text("{}\n")

    assert endpoint_path(training, "greedy") == nested
    assert endpoint_path(training, "stochastic") == stochastic

    base = tmp_path / "base"
    base.mkdir()
    greedy = base / "greedy.scores.jsonl"
    greedy.write_text("{}\n")
    (base / "stochastic_n5.scores.jsonl").write_text("{}\n")
    assert endpoint_path(base, "greedy") == greedy


def test_endpoint_path_rejects_ambiguous_runtime_outputs(tmp_path):
    for runtime in ("debug_a", "debug_b"):
        path = tmp_path / runtime / "eval_results" / "301.scores.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    with pytest.raises(AssertionError, match="expected one greedy endpoint"):
        endpoint_path(tmp_path, "greedy")


def test_transfer_seed_is_shared_across_models_and_output_views():
    base = 20_260_818
    assert transfer_bootstrap_seed(base, "coin_city", "direct_a") == base
    assert transfer_bootstrap_seed(base, "coin_harbor", "direct_a") == base + 1
    assert transfer_bootstrap_seed(base, "coin_city", "mediated_b") == base + 2
    assert transfer_bootstrap_seed(base, "coin_harbor", "mediated_b") == base + 3


def test_structure_a_is_direct_and_has_no_delayed_pulse():
    parameters = {"gain": 0.7, "rho": 0.0, "phi": 0.0, "lambda": 1.0}
    values = forecast_scenarios(COIN_CITY, "direct_a", [], parameters)
    response = response_vector(values)
    assert np.max(np.abs(response[:4])) > 1.0
    assert np.allclose(response[4:], 0.0)


def test_structure_b_has_persistent_delayed_response():
    parameters = {"gain": 0.7, "rho": 0.6, "phi": 0.3, "lambda": 0.8}
    values = forecast_scenarios(COIN_HARBOR, "mediated_b", [], parameters)
    response = response_vector(values)
    assert np.max(np.abs(response[:4])) > 1.0
    assert np.max(np.abs(response[4:])) > 1.0


def test_build_manifest_if_present():
    path = ROOT / "data" / "build_manifest.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["train_structures"] == ["direct_a"]
    assert payload["eval_structures"] == ["direct_a", "mediated_b"]
    assert payload["heldout_rows"] == 1440


def test_renderer_uses_frozen_no_hint_label():
    """Keep the paper validator aligned with the dataset's registered cue key."""
    assert {group[2] for group in EXPECTED_GROUPS} == {
        "correct", "none", "misleading"
    }
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for k in (0, 2, 4, 8):
                pair_id = f"eval:coin_harbor:mediated_b:r0:k{k}"
                misleading = {0: 5.0, 2: 4.0, 4: 3.0, 8: 2.0}[k]
                for cue, mae in (("correct", 1.0), ("none", 2.0),
                                 ("misleading", misleading)):
                    rows.append({
                        "model": model,
                        "arm": "causal",
                        "seed": seed,
                        "decode": "stochastic",
                        "domain": "coin_harbor",
                        "target_structure": "mediated_b",
                        "cue": cue,
                        "k": k,
                        "pair_id": pair_id,
                        "task_id": f"{pair_id}:{cue}",
                        "response_mae": mae,
                    })
    rendered = render_cues(rows, repetitions=20, bootstrap_seed=7)
    assert rendered.count("& 1.00 & 2.00 &") == 12
    assert rendered.count("& 1.00 [1.00,1.00] &") == 12
    common = {"model": "qwen3_4b", "arm": "causal", "decode": "stochastic",
              "domain": "coin_harbor", "target_structure": "mediated_b"}
    override = paired_override_interval(rows, common, repetitions=20, seed=7)
    assert override == (-3.0, -3.0, -3.0)


def test_all_registered_estimands_render_with_seed_first_intervals():
    rows = []
    for model_index, model in enumerate(MODELS):
        for decode in ("greedy", "stochastic"):
            for domain in ("coin_city", "coin_harbor"):
                for structure in ("direct_a", "mediated_b"):
                    for seed in SEEDS:
                        for k in (0, 2, 4, 8):
                            for replicate in range(2):
                                pair_id = (
                                    f"eval:{domain}:{structure}:r{replicate}:k{k}"
                                )
                                for cue in ("correct", "none", "misleading"):
                                    cue_penalty = {
                                        "correct": 0.0,
                                        "none": 1.0,
                                        "misleading": 4.0 - 0.375 * k,
                                    }[cue]
                                    for arm, arm_penalty in (
                                        ("causal", 0.0), ("population_prior", 2.0)
                                    ):
                                        rows.append({
                                            "model": model, "arm": arm, "seed": seed,
                                            "decode": decode, "domain": domain,
                                            "target_structure": structure, "cue": cue,
                                            "k": k, "pair_id": pair_id,
                                            "task_id": f"{pair_id}:cue-{cue}",
                                            "response_mae": (
                                                1.0 + model_index + arm_penalty + cue_penalty
                                            ),
                                        })
                                    if model == "qwen3_4b":
                                        rows.append({
                                            "model": model, "arm": "structureless",
                                            "seed": seed, "decode": decode, "domain": domain,
                                            "target_structure": structure, "cue": cue,
                                            "k": k, "pair_id": pair_id,
                                            "task_id": f"{pair_id}:cue-{cue}",
                                            "response_mae": 6.0 + cue_penalty,
                                        })
                                rows.append({
                                    "model": model, "arm": "base", "seed": 0,
                                    "decode": decode, "domain": domain,
                                    "target_structure": structure, "cue": "correct",
                                    "k": k, "pair_id": pair_id,
                                    "task_id": f"{pair_id}:cue-correct",
                                    "response_mae": 5.0 + model_index,
                                })
    diagnostics = render_diagnostics(rows, repetitions=20, bootstrap_seed=11)
    primary = render_primary(rows, repetitions=20, bootstrap_seed=11)
    summary = render_summary(rows, repetitions=20, bootstrap_seed=11)
    estimates = registered_estimates(rows, repetitions=20, bootstrap_seed=11)
    assert diagnostics.count("[2.00,2.00]") == 12
    assert APPENDIX_MODELS == ("qwen3_4b", "llama3_1_8b")
    assert "Qwen3-4B" in primary and "Llama-3.1-8B" in primary
    assert "Qwen3-8B" not in primary
    assert "Greedy" not in primary and "Decode" not in primary
    assert "absent-minus-correct" in summary
    assert "$k=8$ minus $k=0$ change" in summary
    assert len(estimates["primary"]) == 6
    assert len(estimates["transfer_cells"]) == 12
    assert len(estimates["cue_by_k"]) == 12
    assert len(estimates["cue_overall"]) == 3
    assert len(estimates["evidence_override"]) == 3
    assert {row["estimate"] for row in estimates["evidence_override"]} == {-3.0}
