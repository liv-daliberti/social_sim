"""Regression checks tying main-paper numerical claims to frozen artifacts."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def close(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)


def compact(relative: str) -> str:
    return " ".join((ROOT / relative).read_text().split())


def test_figure1_probe_means_recompute_from_raw_calls() -> None:
    path = ROOT / (
        "exp2_simulated_worlds/biased_news/data/coin_probe/"
        "coin_probe_2026-06-30.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if not row.get("_meta")]
    expected = {
        "gpt-5.4": 0.64375,
        "claude-opus-4-8": 0.68125,
        "DeepSeek-V4-Pro": 0.7125,
    }
    assert len(rows) == 24
    for model, mean in expected.items():
        values = [row["p_heads"] for row in rows if row["model"] == model]
        assert len(values) == 8
        close(statistics.mean(values), mean)

    source = compact("paper/frontmatter.tex")
    assert "$.644$, $.681$, and $.712$" in source


def test_experiment1_headline_values_match_clustered_artifacts() -> None:
    clustered = load_json(
        "exp1_prospective/data/results/clustered_movement_uncertainty.json"
    )["per_model"]
    thresholds = load_json("exp1_prospective/data/results/threshold_robustness.json")[
        "per_model"
    ]
    models = (
        "claude-opus-4-8",
        "gpt-5.4",
        "DeepSeek-V4-Pro",
        "llama3.3:70b",
        "llama3.1:70b",
        "qwen2.5:72b",
        "qwen2.5:32b",
        "qwen2.5:14b",
        "llama3.1:8b",
    )

    n_updates = sum(
        clustered[model]["absolute_revision"][direction]["n_records"]
        for model in models
        for direction in ("pro_H1", "anti_H1", "orthogonal")
    )
    assert n_updates == 30_094

    llama = clustered["llama3.1:8b"]["sensitivity_ratio"]
    claude = clustered["claude-opus-4-8"]["sensitivity_ratio"]
    close(llama["estimate"], 1.9455053847945423)
    close(llama["ci_low"], 1.8397096589764088)
    close(llama["ci_high"], 2.057986762547885)
    close(claude["estimate"], 24.03626290200292)
    close(claude["ci_low"], 17.21112195569617)
    close(claude["ci_high"], 36.324802728462835)
    assert all(clustered[model]["sensitivity_ratio"]["ci_low"] > 1 for model in models)

    ehc3 = [
        thresholds[model]["EHC"]["thresholds"]["3"][
            "market_macro_directional_correctness"
        ]
        for model in models
    ]
    close(min(ehc3), 0.8874909688013136)
    close(max(ehc3), 0.9903234483694254)

    frontier = ("claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro")
    unconditional = [
        thresholds[model]["EHC"]["thresholds"]["0"][
            "market_macro_directional_correctness"
        ]
        for model in frontier
    ]
    omitted = [thresholds[model]["EHC"]["below_3pp_fraction"] for model in frontier]
    assert round(100 * min(unconditional), 1) == 97.3
    assert round(100 * max(unconditional), 1) == 98.4
    assert round(100 * min(omitted), 1) == 6.9
    assert round(100 * max(omitted), 1) == 18.1

    source = compact("paper/experiment1_section.tex")
    for claim in (
        "30,094 valid update records",
        "$1.95$ $[1.84,2.06]$",
        "$24.04$ $[17.21,36.32]$",
        "conditional EHC ranges from .887 to .990",
        "97.3--98.4\\%",
        "6.9--18.1\\%",
    ):
        assert claim in source
    assert "$1.9$--$24\\times$ farther" in compact("paper/frontmatter.tex")


def test_experiment1_human_review_values_match_frozen_export() -> None:
    final = load_json(
        "exp1_prospective/stage3_materials_annotation/data/exports/"
        "final_summary_current.json"
    )
    assert final["analysis"] == "final_descriptive_materials_review"
    assert final["analysis_status"] == "complete"
    assert final["collection"] == {
        "closed": True,
        "completed_reviewer_count": 8,
        "included_reviewer_count": 7,
        "excluded_reviewer_count": 1,
        "minimum_included_reviewers": 6,
        "descriptive_target_met": True,
    }
    assert final["excluded_reviewers"] == ["annotator_02"]
    assert len(final["included_completed_reviewers"]) == 7
    assert final["pooled_included"]["direction_correct_n"] == 96
    assert final["pooled_included"]["direction_denom"] == 126
    assert final["all_completed_sensitivity"]["direction_correct_n"] == 103
    assert final["all_completed_sensitivity"]["direction_denom"] == 144
    excluded = final["reviewers"]["annotator_02"]
    assert excluded["quality_excluded"] is True
    assert excluded["direction_correct_n"] == 7
    assert excluded["direction_response_pattern"] == {
        "more_likely": 17,
        "no_material_effect": 1,
    }
    for obsolete in ("final_gate_available", "reason_final_gate_unavailable", "status"):
        assert obsolete not in final

    agreement = load_json(
        "exp1_prospective/stage3_materials_annotation/data/exports/"
        "agreement_current.json"
    )
    human = agreement["human_key"]
    assert agreement["n_reviewers"] == 7
    assert human["pooled_correct"] == 96
    assert human["pooled_denom"] == 126
    close(human["majority_exact"], 16 / 18)
    close(human["majority_kappa"], 0.8421052631578947)
    classes = human["by_packet_class"]
    assert classes["pro_H1"]["reviewer_correct"] == 29
    assert classes["anti_H1"]["reviewer_correct"] == 33
    assert classes["orthogonal"]["reviewer_correct"] == 34

    # One-sided exact binomial tests against three-option directional guessing.
    for row in human["per_reviewer"].values():
        correct = row["correct"]
        p_value = sum(
            math.comb(18, k) * (1 / 3) ** k * (2 / 3) ** (18 - k)
            for k in range(correct, 19)
        )
        assert p_value < 0.05

    source = compact("paper/methods_section.tex")
    for claim in (
        "96/126 judgments (76.2\\%)",
        "16/18 items (88.9\\%; Cohen's $\\kappa=.84$)",
        "eight completers; one disclosed post-inspection response-pattern exclusion",
    ):
        assert claim in source
    appendix = compact("paper/experiment1_appendix.tex")
    for claim in (
        "Eight adult reviewers completed the 18-item materials protocol",
        "leaving seven quality-eligible reviewers",
        "29/42 for pro-$H_1$, 33/42 for anti-$H_1$, and 34/42",
        "retaining R2 gives 103/144 (71.5\\%) exact agreement",
    ):
        assert claim in appendix


def test_coin_city_misleading_endpoints_are_complete_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    from paper import generate_coin_city_split_figures as generator

    for model in generator.MODELS:
        assert len(generator.load_wrong_context_responses(model)) == 1_250

    monkeypatch.setattr(generator, "RESPONSES", tmp_path)
    with pytest.raises(RuntimeError, match="missing misleading-arm endpoint"):
        generator.load_wrong_context_responses(generator.MODELS[0])


def test_experiment2_main_ranges_match_frozen_results() -> None:
    result = load_json(
        "exp2_v2/biased_news/data/"
        "coin_city_stable_relationship_claude_n250_v4/analysis/"
        "multimodel_results.json"
    )
    assert result["complete"] is True
    assert len(result["models"]) == 6

    models = result["models"].values()
    k0 = [row["curves"]["0"] for row in models]
    improvements = [
        curve["arms"]["abc_no_context"]["forecast_mae"]
        - curve["arms"]["abc_context"]["forecast_mae"]
        for curve in k0
    ]
    assert round(min(improvements), 2) == 0.37
    assert round(max(improvements), 2) == 0.92
    no_context = [curve["arms"]["abc_no_context"]["forecast_mae"] for curve in k0]
    context = [curve["arms"]["abc_context"]["forecast_mae"] for curve in k0]
    assert round(statistics.median(no_context), 2) == 2.59
    assert round(statistics.median(context), 2) == 1.85
    context_rho = [curve["arms"]["abc_context"]["rho"] for curve in k0]
    no_context_rho = [curve["arms"]["abc_no_context"]["rho"] for curve in k0]
    assert round(min(context_rho), 2) == 0.52
    assert round(max(context_rho), 2) == 0.70
    assert (
        round(min(no_context_rho), 2) == -0.06 or round(min(no_context_rho), 2) == -0.07
    )
    assert round(max(no_context_rho), 2) == 0.08

    k4 = [row["curves"]["4"] for row in result["models"].values()]
    remaining_advantage = [
        curve["arms"]["abc_no_context"]["forecast_mae"]
        - curve["arms"]["abc_context"]["forecast_mae"]
        for curve in k4
    ]
    assert round(min(remaining_advantage), 2) == 0.12
    assert round(max(remaining_advantage), 2) == 0.23

    symbol_results = load_json(
        "exp2_v2/biased_news/data/"
        "coin_city_stable_relationship_claude_n250_v4/analysis/"
        "symbol_context_model_comparison_20260824.json"
    )
    expected_symbol_models = {
        "Qwen3-4B-Instruct-2507",
        "Qwen2.5-7B-Instruct",
        "Qwen3-8B",
        "Qwen2.5-14B-Instruct",
        "Qwen3-14B",
        "Qwen3-32B",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V4-Pro",
        "FW-Kimi-K3",
        "claude-opus-4-8",
        "claude-opus-5",
        "gpt-5.6-sol",
    }
    assert symbol_results["status"] == "complete_posthoc_comparison"
    assert symbol_results["not_for_paper"] is False
    assert set(symbol_results["models"]) == expected_symbol_models
    for result in symbol_results["models"].values():
        assert result["record_counts"] == {
            "abc_context": 1_250,
            "abc_no_context": 1_250,
            "abc_symbol_context": 1_250,
        }

    symbol_models = symbol_results["models"]
    deepseek = symbol_models["DeepSeek-V4-Pro"]["curves"]["0"]
    close(deepseek["arms"]["abc_no_context"]["rho"], 0.017955582400171515)
    close(deepseek["arms"]["abc_symbol_context"]["rho"], 0.45433182016359375)
    rho = deepseek["paired_discrimination"]["symbol_minus_no_context_rho"]
    close(rho["difference"], 0.4363762377634222)
    accuracy = deepseek["paired_discrimination"][
        "symbol_minus_no_context_regime_accuracy"
    ]
    close(accuracy["difference"], 0.172)

    # The matched Qwen3 series crosses from null intervals at 4B/8B to a
    # positive discrimination interval at 14B. This is a family-specific
    # scaling result, not a universal parameter cutoff.
    for model in (
        "Qwen3-4B-Instruct-2507",
        "Qwen3-8B",
    ):
        low, high = symbol_models[model]["curves"]["0"]["paired_discrimination"][
            "symbol_minus_no_context_rho"
        ]["ci_95"]
        assert low < 0 < high
    qwen14_rho = symbol_models["Qwen3-14B"]["curves"]["0"]["paired_discrimination"][
        "symbol_minus_no_context_rho"
    ]
    assert qwen14_rho["difference"] > 0
    assert qwen14_rho["ci_95"][0] > 0

    llama70 = symbol_models["Llama-3.1-70B-Instruct"]["curves"]["0"]
    llama70_rho = llama70["paired_discrimination"]["symbol_minus_no_context_rho"]
    close(llama70_rho["difference"], 0.4238620422170436)
    close(llama70_rho["ci_95"][0], 0.31449774490666854)
    close(llama70_rho["ci_95"][1], 0.5425014691104199)
    assert llama70_rho["ci_95"][0] > 0
    llama70_accuracy = llama70["paired_discrimination"][
        "symbol_minus_no_context_regime_accuracy"
    ]
    close(llama70_accuracy["difference"], 0.092)
    assert llama70_accuracy["ci_95"][0] > 0
    llama70_mae = llama70["paired_forecast_mae"]["symbol_minus_no_context"]
    close(llama70_mae["mean"], 0.4483310883847305)
    assert llama70_mae["ci_95"][0] > 0

    kimi = symbol_models["FW-Kimi-K3"]["curves"]["0"]
    kimi_rho = kimi["paired_discrimination"]["symbol_minus_no_context_rho"]
    close(kimi_rho["difference"], 0.633030521080501)
    close(kimi_rho["ci_95"][0], 0.5196448902038018)
    close(kimi_rho["ci_95"][1], 0.7419758050630081)
    kimi_mae = kimi["paired_forecast_mae"]["symbol_minus_no_context"]
    close(kimi_mae["mean"], -0.49742812660290125)
    assert kimi_mae["ci_95"][1] < 0

    # Kimi, both Opus deployments, and GPT recover the mapping and improve the
    # numerical forecast, unlike the Qwen2.5 checkpoints' positive-Delta-MAE
    # pattern. Mapping induction and calibrated use are separate outcomes.
    for model in (
        "FW-Kimi-K3",
        "claude-opus-4-8",
        "claude-opus-5",
        "gpt-5.6-sol",
    ):
        curve = symbol_models[model]["curves"]["0"]
        assert (
            curve["paired_discrimination"]["symbol_minus_no_context_rho"]["ci_95"][0]
            > 0
        )
        assert curve["paired_forecast_mae"]["symbol_minus_no_context"]["ci_95"][1] < 0
    for model in (
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen3-32B",
    ):
        curve = symbol_models[model]["curves"]["0"]
        assert (
            curve["paired_discrimination"]["symbol_minus_no_context_rho"]["ci_95"][0]
            > 0
        )
        assert curve["paired_forecast_mae"]["symbol_minus_no_context"]["mean"] > 0

    result_table = compact("paper/tables/exp2_symbol_context_results.tex")
    for model in (
        "Qwen3-4B",
        "Qwen2.5-7B",
        "Qwen3-8B",
        "Qwen2.5-14B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Qwen2.5-32B",
        "Qwen2.5-72B",
        "Llama~3.1 8B",
        "Llama~3.1 70B",
        "DeepSeek V4-Pro",
        "Kimi K3",
        "Claude Opus~4.8",
        "Claude Opus~5",
        "GPT-5.6 Sol",
    ):
        assert model in result_table
    scaling_table = compact("paper/tables/exp2_symbol_context_scaling.tex")
    for model in (
        "Qwen2.5-32B",
        "Qwen2.5-72B",
        "Llama~3.1 8B",
        "Llama~3.1 70B",
    ):
        assert model in scaling_table
    appendix = compact("paper/experiment2_appendix.tex")
    assert r"\input{tables/exp2_symbol_context_results}" in appendix
    assert r"\input{tables/exp2_symbol_context_scaling}" in appendix

    source = compact("paper/experiment2_section.tex")
    # Numbers the section states, each traceable to a frozen authority.
    for claim in (
        "$0.37$ to $0.92$",
        "$0.12$--$0.23$",
        "matches or beats a context-blind OLS benchmark",
        "$.51$--$.86$",
        "$.61$--$.90$",
        "Twelve of fifteen systems recover it",
        "within both Qwen3 and Llama only the larger",
        "rather than tracking\nparameter count".replace("\n", " "),
        "input embeddings alone already reach $1.000$",
        "that control falls to $.555$",
        "follows the label at\n$k=0$ and the truth at $k=4$".replace("\n"," "),
        "only the larger\ncheckpoints do".replace("\n"," "),
        "$.333$ at $k=0$",
        r"Appendix~\ref{app:coin-city-reference-selection}",
        r"Appendix~\ref{app:coin-city-mechanistic-probe}",
    ):
        assert claim in source, claim
    # Per-deployment numbers belong in the figure table, not the prose.
    for row in (
        r"\micon{claude.png}~Opus~4.8 & 250 & 2.32",
        r"\micon{openai.png}~GPT-5.6 & 250 & 3.60",
        r"\micon{gemini.png}~3.6 Flash & 250 & 2.53",
    ):
        assert row in source, row
    assert "Forecast MAE [95\\% CI]" in source
    # Both main-text figures must stay: the roster table and the four-arm pattern.
    assert "figures/exp2_coin_city_results.pdf" in source
    assert "figures/exp2_symbol_decoding.pdf" in source
    # The reference-selection figure lives with its tables in the appendix.
    assert "figures/exp2_reference_selection.pdf" not in source
    appendix = compact("paper/experiment2_appendix.tex")
    assert "figures/exp2_reference_selection.pdf" in appendix
    assert "Llama~3.1-8B is the\nnegative control".replace("\n", " ") in source
    assert "A prior attached to the words" in appendix

    # The three structural claims the section is organised around.
    for heading in (
        "Selection and revision",
        "The cue selects a displayed reference, not a prior on its words",
        "Induction of a novel mapping",
    ):
        assert heading in source, heading
    # Scope statements that must survive any future tightening of this section.
    for bound in (
        "measure how a stated assignment becomes a number",
        "selection is a two-way choice",
        "exact within-regime coefficient is never recovered",
        "the causal test on one",
    ):
        assert bound in source, bound
    for stale_claim in (
        "all three Qwen2.5 checkpoints",
        "provide converging representational evidence",
        "Hidden states track the selected relationship",
        "compared with $-.07$ to $.08$ without context",
        "Novel cue induction varies across models and families",
        "smaller matched counterparts",
        "episode-specific within-regime residual",
    ):
        assert stale_claim not in source, stale_claim
    assert "Sufficiently large models" not in source
    assert "Episode-local cue induction is model-dependent" not in source
    assert "DeepSeek, but not Qwen3-4B" not in source
    assert "establish reference selection" not in source

    front = compact("paper/frontmatter.tex")
    assert "target observations weaken its effect" in front
    assert "Sufficiently large models also recover arbitrary mappings" in front
    assert "Ten of 13 systems recover" not in front
    assert "Sufficiently large open-weight and hosted models" not in front
    assert "family-specific rather than a universal size threshold" not in front
    assert "episode-randomized symbol control" not in front
    assert "observations override it" not in front
    conclusion = compact("paper/conclusion.tex")
    assert "arbitrary-label control" not in conclusion
    for claim in (
        "central result is behavioral, not a claim of an explicit world model",
        "forecasts behave as if selecting a context-dependent regime",
        "decode the cue-selected response family but not within-regime coefficient variation",
    ):
        assert claim in conclusion


def test_experiment3_main_values_match_registered_analysis() -> None:
    protocol = load_json(
        "exp3_training_transfer/coin_city_structural/protocol/"
        "coin_city_structural_manifest.json"
    )
    assert protocol["status"] == "passed"
    for check in (
        "causal_prior_prompts_byte_identical",
        "population_prior_exact_target_marginal",
        "population_prior_no_fixed_reward_links",
        "causal_rewards_are_episode_truth",
        "training_contains_only_structure_a",
        "structure_b_language_absent_from_training",
        "heldout_byte_identical_across_training_arms",
        "disjoint_train_eval_identifiers",
    ):
        assert protocol["checks"][check] is True

    registered = load_json(
        "exp3_training_transfer/coin_city_structural/reports/registered_results.json"
    )
    assert registered["validated_jobs"] == 24
    assert registered["validated_tasks_per_job"] == 1_440
    assert registered["row_draws"] == 207_360
    score_files = [
        path if path.is_absolute() else ROOT / path
        for path in map(Path, registered["score_files"])
    ]
    stochastic_files = [path for path in score_files if "stochastic_n5" in path.name]
    greedy_files = [path for path in score_files if "stochastic_n5" not in path.name]
    assert len(stochastic_files) == len(greedy_files) == 24
    stochastic_total = 0
    stochastic_parsed = 0
    endpoint_rates = []
    for path in stochastic_files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(rows) == 7_200
        parsed = sum(bool(row["parsed"]) for row in rows)
        stochastic_total += len(rows)
        stochastic_parsed += parsed
        endpoint_rates.append(parsed / len(rows))
    assert stochastic_total == 172_800
    assert stochastic_parsed == 172_732
    assert min(endpoint_rates) >= 0.99819

    result = registered["registered_estimates"]
    qwen8 = next(
        row
        for row in result["primary"]
        if row["model"] == "qwen3_8b" and row["decode"] == "stochastic"
    )
    close(qwen8["mean_response_mae"]["base"], 5.7176971283333335)
    close(qwen8["mean_response_mae"]["population_prior"], 4.708329330972223)
    close(qwen8["mean_response_mae"]["causal"], 4.457035298888889)
    contrast = qwen8["population_prior_minus_causal"]
    close(contrast["estimate"], 0.2512940320833334)
    close(contrast["ci95_low"], 0.058208621503472203)
    close(contrast["ci95_high"], 0.46998328662500005)

    qwen8_cells = [
        row for row in result["transfer_cells"] if row["model"] == "qwen3_8b"
    ]
    assert len(qwen8_cells) == 4
    assert sum(row["ci95_low"] > 0 for row in qwen8_cells) == 3
    structure_only = next(
        row
        for row in qwen8_cells
        if row["domain"] == "coin_city" and row["target_structure"] == "mediated_b"
    )
    assert structure_only["ci95_low"] < 0 < structure_only["ci95_high"]

    cues = next(row for row in result["cue_overall"] if row["model"] == "qwen3_8b")
    close(cues["absent_minus_correct"]["estimate"], 0.11696011833333332)
    close(cues["misleading_minus_correct"]["estimate"], 0.3086490208333334)
    qwen4 = next(
        row
        for row in result["primary"]
        if row["model"] == "qwen3_4b" and row["decode"] == "stochastic"
    )
    assert qwen4["population_prior_minus_causal"]["ci95_low"] > 0

    llama = next(
        row
        for row in result["primary"]
        if row["model"] == "llama3_1_8b" and row["decode"] == "stochastic"
    )
    close(llama["mean_response_mae"]["base"], 9.42750767125)
    close(llama["mean_response_mae"]["population_prior"], 4.687181567777778)
    close(llama["mean_response_mae"]["causal"], 4.267135969861111)
    llama_contrast = llama["population_prior_minus_causal"]
    close(llama_contrast["estimate"], 0.4200455979166667)
    assert llama_contrast["ci95_low"] < 0 < llama_contrast["ci95_high"]
    llama_cues = next(
        row for row in result["cue_overall"] if row["model"] == "llama3_1_8b"
    )
    close(llama_cues["absent_minus_correct"]["estimate"], 0.2188010929166666)
    close(llama_cues["absent_minus_correct"]["ci95_low"], 0.12323785587847232)
    close(llama_cues["misleading_minus_correct"]["estimate"], 0.20964923416666684)
    close(llama_cues["misleading_minus_correct"]["ci95_low"], 0.09205618988888888)
    llama_override = next(
        row for row in result["evidence_override"] if row["model"] == "llama3_1_8b"
    )
    close(llama_override["estimate"], -0.6210503149999999)
    close(llama_override["ci95_high"], -0.3228456566250002)

    source = compact("paper/experiment3_section.tex")
    for claim in (
        "response MAE is $5.72$ for the untrained base, $4.71$ for population-prior training, and $4.46$",
        "$.251$ $[.058,.470]$",
        "$1.26$ points (22\\%)",
        "only the prompt--key pairing differs",
        "learning how each episode's evidence maps to outcomes",
        "$4.46$, $4.57$, and $4.77$",
        "$.117$ $[.010,.230]$",
        "$.309$ $[.206,.411]$",
    ):
        assert claim in source
    assert (
        "episode-matched response MAE is $.25$ $[.06,.47]$ lower than the population-prior control"
        in compact("paper/frontmatter.tex")
    )
    methods = compact("paper/methods_section.tex")
    assert (
        "The episode-matched and population-prior arms use byte-identical prompts and the "
        "same multiset of answer keys." in methods
    )
    setup = compact("paper/experiment3_training_setup_appendix.tex")
    assert "($42,43,44$) $=18$ full training runs." in setup
    assert "$=12$ full training runs." not in setup
    appendix = compact("paper/experiment3_appendix.tex")
    for claim in (
        "all 24 endpoints and 172,800 stochastic rows",
        "172,732 parse under the strict contract (99.96\\%)",
        "at least 99.82\\% coverage",
        "The prior-minus-episode-matched training contrast is positive, $.420$",
        "$[-.115,.958]$",
        "absent-minus-correct MAE is $.219$ $[.123,.313]$",
        "misleading-minus-correct is $.210$ $[.092,.320]$",
        "$-.621$ $[-.957,-.323]$",
    ):
        assert claim in appendix


def test_experiment4_main_values_match_locked_test() -> None:
    manifest = load_json(
        "exp3_training_transfer/polymarket/data/exp3b_registered/manifest.json"
    )
    preflight = load_json(
        "exp3_training_transfer/polymarket/protocol/exp3b_preflight.json"
    )
    expected_counts = {"train": 1_736, "dev": 512, "test": 1_024}
    assert {
        split: manifest["selected"][split]["count"] for split in expected_counts
    } == expected_counts
    assert manifest["cross_split_key_overlap"] == {
        "train_dev": 0,
        "train_test": 0,
        "dev_test": 0,
    }
    assert manifest["design"]["cross_split_exclusions"] == [
        "event_id",
        "series_id",
        "normalized_question_template",
    ]
    assert preflight["status"] == "pass"
    assert preflight["counts"] == expected_counts
    assert preflight["cross_split_key_overlap"] == manifest["cross_split_key_overlap"]
    assert preflight["training_evaluation_split"] == "dev"
    assert preflight["locked_final_split"] == "test"

    summaries = {
        "qwen3_4b": load_json(
            "exp3_training_transfer/polymarket/reports/"
            "exp3b_locked_test_j30505540.summary.json"
        ),
        "qwen3_8b": load_json(
            "exp3_training_transfer/polymarket/reports/"
            "exp3b_qwen3_8b_locked_test_j30856250.summary.json"
        ),
        "llama3_1_8b": load_json(
            "exp3_training_transfer/polymarket/reports/"
            "exp3b_llama3_1_8b_locked_test_j30856254.summary.json"
        ),
    }
    endpoint = summaries["qwen3_8b"]
    assert endpoint["models"]["base"]["n"] == 1_024
    assert all(row["parse_coverage"] == 1 for row in endpoint["models"].values())
    close(endpoint["models"]["base"]["brier"], 0.12866662963871944)
    trained_brier = statistics.mean(
        endpoint["models"][f"seed_{seed}"]["brier"] for seed in (42, 43, 44)
    )
    close(trained_brier, 0.11641605647787746)
    comparison = endpoint["comparisons_brier"]["trained_seed_mean_minus_base"]
    close(comparison["estimate"], -0.01225057316084212)
    close(comparison["ci95_low"], -0.01948130056448668)
    close(comparison["ci95_high"], -0.006019195538087603)
    assert all(
        endpoint["models"][f"seed_{seed}"]["brier"]
        < endpoint["models"]["base"]["brier"]
        for seed in (42, 43, 44)
    )
    close(endpoint["models"]["base"]["log_loss_nats"], 0.5003233749857193)
    trained_log_loss = statistics.mean(
        endpoint["models"][f"seed_{seed}"]["log_loss_nats"] for seed in (42, 43, 44)
    )
    close(trained_log_loss, 0.4090920161877383)
    close(endpoint["baselines"]["market"]["brier"], 0.11369354125976557)
    close(
        endpoint["baselines"]["platt_market_train_only"]["brier"],
        0.11232485790149636,
    )
    for summary in summaries.values():
        base_comparison = summary["comparisons_brier"]["trained_seed_mean_minus_base"]
        assert base_comparison["estimate"] < 0
        assert base_comparison["ci95_high"] < 0
        market_comparison = summary["comparisons_brier"][
            "trained_seed_mean_minus_market"
        ]
        assert market_comparison["ci95_low"] <= 0 <= market_comparison["ci95_high"]
    for model_key in ("qwen3_8b", "llama3_1_8b"):
        platt_comparison = summaries[model_key]["comparisons_brier"][
            "trained_seed_mean_minus_platt_market"
        ]
        assert platt_comparison["ci95_low"] > 0
    llama = summaries["llama3_1_8b"]
    close(llama["models"]["base"]["parse_coverage"], 1012 / 1024)
    assert all(
        llama["models"][f"seed_{seed}"]["parse_coverage"] == 1 for seed in (42, 43, 44)
    )

    movement = load_json(
        "exp3_training_transfer/polymarket/reports/"
        "exp3b_exp1_reapplication_j30535058/movement_component_analysis.json"
    )["comparisons"]["trained_mean_minus_base_EHC"]
    close(movement["estimate"], 0.01011111111111111)
    close(movement["ci95_low"], 0.0004444444444444462)
    close(movement["ci95_high"], 0.020666666666666663)

    scale = load_json(
        "exp3_training_transfer/polymarket/reports/exp4_scale_latest.summary.json"
    )
    assert scale["status"] == "pass"
    assert scale["protocol_version"] == "exp4_qwen_scale_v1"
    assert scale["n"] == 318
    assert scale["decoding"]["mode"] == "five_draw_non_thinking_stochastic"
    assert set(scale["models"]) == {"qwen3_4b", "qwen3_8b", "qwen3_14b"}
    close(scale["models"]["qwen3_4b"]["trained_seed_mean_brier"], 0.13664273538365726)
    close(scale["models"]["qwen3_8b"]["trained_seed_mean_brier"], 0.12858730072328203)
    close(scale["models"]["qwen3_14b"]["trained_seed_mean_brier"], 0.1269518425262066)
    scale_contrast = scale["cross_model_brier"]["qwen3_14b_minus_qwen3_8b"]
    close(scale_contrast["estimate"], -0.0016354581970754708)
    close(scale_contrast["ci95_low"], -0.0033189760086375386)
    close(scale_contrast["ci95_high"], -0.00024060183652547788)
    assert scale["decision_flags"]["qwen3_14b_beats_qwen3_8b_conclusive"] is True
    assert scale["decision_flags"]["qwen3_14b_beats_market_conclusive"] is False
    assert (
        scale["decision_flags"]["qwen3_14b_beats_train_only_platt_conclusive"] is False
    )
    close(scale["baselines"]["market"]["brier"], 0.12654794182389942)
    crowd_contrast = scale["qwen3_14b_trained_minus_market"]
    close(crowd_contrast["estimate"], 0.0004039007023071293)
    close(crowd_contrast["ci95_low"], -0.00027547354297484427)
    close(crowd_contrast["ci95_high"], 0.0013911306120218579)

    scale_llama = load_json(
        "exp3_training_transfer/polymarket/reports/"
        "exp4_scale_llama3_1_8b_stochastic_j30889436.summary.json"
    )
    assert scale_llama["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert scale_llama["model_key"] == "llama3_1_8b"
    assert scale_llama["holdout"]["n"] == 318
    close(scale_llama["models"]["base"]["brier"], 0.16417377306307376)
    close(scale_llama["models"]["base"]["draw_parse_coverage"], 1581 / 1590)
    close(
        sum(scale_llama["models"][f"seed_{seed}"]["brier"] for seed in (42, 43, 44))
        / 3,
        0.12642974319706817,
    )
    llama_crowd = scale_llama["comparisons_brier"]["trained_seed_mean_minus_market"]
    close(llama_crowd["estimate"], -0.00011819862683123503)
    close(llama_crowd["ci95_low"], -0.0004550035753476317)
    close(llama_crowd["ci95_high"], 0.00011862905623393738)
    llama_ledger = load_json(
        "exp3_training_transfer/polymarket/runs/"
        "exp4_scale_llama_posthoc_20260825T201141Z.json"
    )
    assert llama_ledger["status"] == "complete"
    assert llama_ledger["classification"] == "post_hoc_architecture_comparator"

    ensemble = load_json(
        "exp3_training_transfer/polymarket/reports/"
        "exp4_hosted_ensemble_posthoc.summary.json"
    )
    available = ensemble["results"]["available_member_pool_all_tasks"]
    close(available["brier"], 0.1273034985082547)
    close(available["ensemble_minus_market"]["estimate"], 0.0007555566843553481)
    close(available["ensemble_minus_market"]["ci95_low"], -0.0006369892656040217)
    close(available["ensemble_minus_market"]["ci95_high"], 0.00220301543161291)
    assert "do not beat" in ensemble["conclusion"]

    source = compact("paper/experiment3_section.tex")
    for claim in (
        "a frozen scale extension uses 318 new markets",
        "reports Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-14B as a connected curve",
        "The post-hoc Qwen3-1.7B trained mean is $.1248$ (base $.2253$)",
        "registered 4B/8B/14B trained means are $.1366$/$.1286$/$.1270$",
        "$-.00117$ $[-.00298,+.00016]$",
        "contemporaneous crowd (Brier $.12655$)",
        "$+.00040$ $[-.00028,+.00139]$",
        "post-hoc Llama minus crowd is $-.00012$ $[-.00046,+.00012]$",
        "the disclosed Kimi repair are tied with users",
        "Opus~5 are worse under fail-closed scoring",
    ):
        assert claim in source
    assert "train-only Platt" not in source
    assert "14B-minus-8B" not in source
    assert "same-release" not in source
    assert (
        "On a new 318-market holdout, trained Brier falls from $.1366$ to $.1286$ to $.1270$"
        in compact("paper/frontmatter.tex")
    )
    assert (
        r"\includegraphics[width=\linewidth]{figures/exp4_qwen_scale_results.pdf}"
        in source
    )
    assert r"\input{tables/exp3b_qwen3_8b_endpoint_results}" not in source
    assert "exp3b_endpoint_results" not in source
    methods = compact("paper/methods_section.tex")
    assert (
        "Calendar splits contain 1,736 training, 512 development, and 1,024 test markets, "
        "with no event or normalized question family shared across splits." in methods
    )
    assert "for 300 rollout-and-update rounds (4,800 prompt presentations)" in methods
    assert "for 300 updates" not in methods
    appendix = compact("paper/experiment3_appendix.tex")
    assert "latest pre-cutoff market price obtains Brier $.11369$" in appendix
    assert "training split obtains $.11232$" in appendix
    assert r"\input{tables/exp3b_model_roster_results}" in appendix
    assert "Llama-8B improves from $.15203$ to $.11502$" in appendix
    assert (
        "base-relative improvement rather than superiority to market-based forecasts"
        in appendix
    )
    assert "Qwen3-4B Evidence-Updating Evaluation" in appendix
    assert r"$+.0101$ with 95\% interval $[+.0004,+.0207]$" in appendix
    assert "temperature-zero JSON decoding" not in source
    assert "temperature-zero result predates the new-holdout stochastic" in appendix
    assert r"\input{tables/exp3b_qwen3_8b_endpoint_results}" in appendix
    assert r"\input{tables/exp4_qwen_scale_results}" in appendix
    assert "same-release 14B-minus-8B trained contrast" in appendix
    assert "later Instruct-2507 revision" in appendix
    assert "All 4,770 trained draws parse" in appendix
    assert "nine of the 1,590 base draws do not" in appendix
    assert "Trained minus base is $-.03774$ $[-.05860,-.01946]$" in appendix
    assert "The repaired result parses all 1,590 draws and has" in appendix
    assert "Brier $.12682$" in appendix
    assert "Fail-closed Brier is $.15571$" in appendix
    assert "pooling does not beat the contemporaneous users" in appendix
