from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "eval"))

from analysis.analyze_coin_city_rationales import (  # noqa: E402
    classify_reference,
    development_episodes,
    semantic_regime,
)
from engine.coin_city_robustness_population import (  # noqa: E402
    CITY_SLOPE_SD,
    EXPERIMENT,
    SEED_BASE,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
    make_episode,
    task_id,
)
from analysis.analyze_coin_city_robustness_population import (  # noqa: E402
    ARMS as POPULATION_ARMS,
    BOOTSTRAP_DRAWS as POPULATION_BOOTSTRAP_DRAWS,
    MODELS as POPULATION_MODELS,
)


def test_robustness_population_is_fresh_and_deterministic() -> None:
    assert EXPERIMENT == "coin_city_population_075_040_sd010_v1"
    assert (STRONG_SLOPE_MEAN, WEAK_SLOPE_MEAN, CITY_SLOPE_SD) == (0.75, 0.40, 0.10)
    assert SEED_BASE == 940_000
    assert make_episode(0) == make_episode(0)
    assert make_episode(0)["seed"] == 940_000
    assert task_id(249, 4) == "coincityrobustv1_0249_c4"


def test_coder_split_is_frozen_and_complete() -> None:
    dev = development_episodes()
    assert len(dev) == 50
    assert len(set(range(250)) - dev) == 200
    expected = set(
        sorted(
            range(250),
            key=lambda e: hashlib.sha256(
                f"rationale-coder-v1:{e}".encode()
            ).hexdigest(),
        )[:50]
    )
    assert dev == expected


def test_reference_coder_prefers_explicit_link_and_fails_closed() -> None:
    assert classify_reference("City C shares KIV with City A, so I use its cases.") == (
        "A",
        "explicit_link",
    )
    assert classify_reference(
        "Using the responsiveness observed in City B gives 42."
    ) == ("B", "explicit_link")
    assert classify_reference("City A is noisy but City B is useful.")[0] is None
    assert classify_reference("KIV cities are more responsive.")[0] is None


def test_semantic_coder_requires_exclusive_regime_word() -> None:
    assert semantic_regime("City C uses national-news coverage.") is True
    assert semantic_regime("City C uses local news coverage.") is False
    assert semantic_regime("National and local cities differ.") is None
    assert semantic_regime("The response is muted.") is None


def test_gpt_repeat_is_isolated_to_registered_k0_arms() -> None:
    path = ROOT / "eval" / "run_coin_city_gpt56_k0_repeat.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"ARMS", "PREFIXES"}:
                literals[target.id] = ast.literal_eval(node.value)
    assert literals["ARMS"] == ("abc_no_context", "abc_context", "abc_symbol_context")
    assert literals["PREFIXES"] == (0,)
    assert "PLANNED_CALLS = shared.EPISODES * len(ARMS)" in source
    assert "if PLANNED_CALLS != 750:" in source


def test_population_analysis_roster_and_bootstrap_are_frozen() -> None:
    assert POPULATION_ARMS == ("abc_no_context", "abc_context", "abc_symbol_context")
    assert len(POPULATION_MODELS) == 10
    assert len({model for _, model in POPULATION_MODELS}) == 10
    assert POPULATION_BOOTSTRAP_DRAWS == 5_000

def test_completed_extension_artifacts_are_internally_consistent() -> None:
    harder_root = ROOT / "data" / "coin_city_population_075_040_sd010_v1"
    primary_root = (
        ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
    )
    population = json.loads(
        (harder_root / "analysis" / "population_robustness_results_v1.json").read_text()
    )
    assert population["status"] == "complete"
    assert population["bootstrap_draws"] == 5_000
    assert population["generator_population"]["realized"][
        "reference_latent_reversals"
    ] == 2
    assert len(population["models"]) == 10
    for result in population["models"].values():
        assert result["record_counts"] == {
            "abc_context": 1_250,
            "abc_no_context": 1_250,
            "abc_symbol_context": 1_250,
        }

    comparisons = population["k0_point_comparison"]

    def positive(model: str, population_name: str) -> bool:
        low, _ = comparisons[model][population_name][
            "symbol_minus_no_context_rho"
        ]["ci_95"]
        return low > 0

    assert sum(positive(model, "original_population") for model in comparisons) == 7
    assert sum(positive(model, "harder_population") for model in comparisons) == 7
    assert not positive("Qwen3-8B", "original_population")
    assert positive("Qwen3-8B", "harder_population")
    assert positive("Qwen3-14B", "original_population")
    assert not positive("Qwen3-14B", "harder_population")
    assert not positive("Llama-3.1-8B-Instruct", "original_population")
    assert not positive("Llama-3.1-8B-Instruct", "harder_population")
    assert positive("Llama-3.1-70B-Instruct", "original_population")
    assert positive("Llama-3.1-70B-Instruct", "harder_population")

    rationales = json.loads(
        (primary_root / "analysis" / "rationale_audit_v1.json").read_text()
    )
    assert rationales["status"] == "complete"
    assert rationales["presence"]["total_records"] == 56_250
    assert rationales["presence"]["total_rationales"] == 56_057
    symbol_audits = [row["audit"] for row in rationales["symbol_k0"].values()]
    assert sum(row["counts"]["records"] for row in symbol_audits) == 3_000
    assert sum(row["counts"]["rationales"] for row in symbol_audits) == 2_992
    assert sum(row["counts"]["reference_codable"] for row in symbol_audits) == 1_832
    assert sum(
        row["counts"]["names_matching_reference"] for row in symbol_audits
    ) == 1_832
    inverted_audits = [
        row["audit"] for row in rationales["inverted_semantic_k0"].values()
    ]
    assert sum(row["counts"]["rationales"] for row in inverted_audits) == 1_200
    assert sum(
        row["counts"]["names_false_cue_reference"] for row in inverted_audits
    ) == 1_089
    assert sum(
        row["counts"]["names_false_cue_regime"] for row in inverted_audits
    ) == 1_186

    repeat = json.loads(
        (primary_root / "analysis" / "gpt56_k0_repeat_v1.json").read_text()
    )
    assert repeat["status"] == "complete"
    assert repeat["jointly_parsed_episodes"] == 250
    assert all(
        validation
        == {
            "records": 250,
            "unique_tasks": 250,
            "parsed": 250,
            "transport_failures": 0,
            "unparseable": 0,
        }
        for validation in repeat["validation"].values()
    )
    symbol_repeat = repeat["paired_arm_effects"][
        "abc_symbol_context_minus_abc_no_context"
    ]
    rho_did = symbol_repeat["repeat_minus_original_rho_effect"]
    assert rho_did["difference_in_differences"] < 0
    assert rho_did["ci_95"][0] < 0 < rho_did["ci_95"][1]
    mae_did = symbol_repeat["repeat_minus_original_forecast_mae"]
    assert mae_did["mean"] > 0
    assert mae_did["ci_95"][0] < 0 < mae_did["ci_95"][1]

