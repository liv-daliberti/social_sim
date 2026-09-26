#!/usr/bin/env python3
"""Build the released 15-model Coin City comparison artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BIASED_NEWS = HERE.parents[1]
ANALYZER_PATH = BIASED_NEWS / "analysis" / "analyze_coin_city_symbol_context.py"
INTERIM_PATH = HERE / "exploratory_results_interim.json"
OUTPUT_PATH = HERE / "exploratory_results.json"
SUMMARY_PATH = HERE / "exploratory_summary.md"
PAPER_OUTPUT_PATH = (
    BIASED_NEWS
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "symbol_context_model_comparison_20260824.json"
)
BOOTSTRAP_DRAWS = 2_000
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")

MODEL_ORDER = (
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
)
DISPLAY = {
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-32B": "Qwen3-32B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "FW-Kimi-K3": "Kimi K3",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-5": "Claude Opus 5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
}
LOCAL_MODEL_DIRS = {
    "Qwen2.5-14B-Instruct": HERE / "qwen2_5_14b",
    "Qwen3-32B": HERE / "qwen3_32b",
    "Qwen2.5-32B-Instruct": HERE / "qwen2_5_32b",
    "Qwen2.5-72B-Instruct": HERE / "qwen2_5_72b",
    "Llama-3.1-8B-Instruct": HERE / "llama_3_1_8b",
    "Llama-3.1-70B-Instruct": HERE / "llama_3_1_70b",
}
PRIMARY_RESPONSES = (
    BIASED_NEWS
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "responses"
)
HOSTED_SYMBOL_MODELS = ("FW-Kimi-K3", "claude-opus-4-8")
ADDED_MODELS = tuple(LOCAL_MODEL_DIRS) + HOSTED_SYMBOL_MODELS


def load_analyzer():
    sys.path.insert(0, str(ANALYZER_PATH.parent))
    spec = importlib.util.spec_from_file_location("coin_city_symbol_analysis", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analyzer from {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response_path(model: str, arm: str) -> Path:
    if model in HOSTED_SYMBOL_MODELS:
        if arm == "abc_symbol_context":
            return HERE / "hosted" / f"responses_{model}_{arm}.jsonl"
        return PRIMARY_RESPONSES / f"responses_{model}_{arm}.jsonl"
    directory = LOCAL_MODEL_DIRS[model]
    reparsed = directory.with_name(directory.name + "_reparsed")
    candidate = reparsed / f"responses_{model}_{arm}.jsonl"
    if candidate.exists():
        return candidate
    return directory / f"responses_{model}_{arm}.jsonl"


def number(value: float) -> str:
    return f"{value:.3f}"


def interval(metric: dict, key: str) -> str:
    low, high = metric["ci_95"]
    return f"{number(metric[key])} [{number(low)}, {number(high)}]"


def primary_row(model: str, result: dict) -> str:
    curve = result["curves"]["0"]
    arms = curve["arms"]
    delta_mae = curve["paired_forecast_mae"]["symbol_minus_no_context"]
    delta_rho = curve["paired_discrimination"]["symbol_minus_no_context_rho"]
    delta_accuracy = curve["paired_discrimination"][
        "symbol_minus_no_context_regime_accuracy"
    ]
    return "| " + " | ".join(
        (
            DISPLAY[model],
            str(curve["n"]),
            number(arms["abc_no_context"]["forecast_mae"]),
            number(arms["abc_context"]["forecast_mae"]),
            number(arms["abc_symbol_context"]["forecast_mae"]),
            interval(delta_mae, "mean"),
            number(arms["abc_no_context"]["rho"]),
            number(arms["abc_symbol_context"]["rho"]),
            interval(delta_rho, "difference"),
            number(arms["abc_no_context"]["regime_accuracy"]),
            number(arms["abc_symbol_context"]["regime_accuracy"]),
            interval(delta_accuracy, "difference"),
        )
    ) + " |"


def scaling_curve_row(model: str, k: int, result: dict) -> str:
    curve = result["curves"][str(k)]
    delta_mae = curve["paired_forecast_mae"]["symbol_minus_no_context"]
    delta_rho = curve["paired_discrimination"]["symbol_minus_no_context_rho"]
    delta_accuracy = curve["paired_discrimination"][
        "symbol_minus_no_context_regime_accuracy"
    ]
    return "| " + " | ".join(
        (
            DISPLAY[model],
            str(k),
            str(curve["n"]),
            interval(delta_mae, "mean"),
            interval(delta_rho, "difference"),
            interval(delta_accuracy, "difference"),
        )
    ) + " |"


def main() -> None:
    interim = json.loads(INTERIM_PATH.read_text(encoding="utf-8"))
    analyzer = load_analyzer()
    scores = {
        row["task_id"]: row
        for row in analyzer.read_jsonl(analyzer.DESIGN / "scoring_key.jsonl")
    }
    episodes = {
        row["episode"]: row
        for row in analyzer.read_jsonl(analyzer.DESIGN / "episodes.jsonl")
    }

    models = dict(interim["models"])
    response_paths = dict(interim["response_paths"])
    analyzer.response_path = response_path
    for model in ADDED_MODELS:
        models[model] = analyzer.analyze_model(
            model,
            scores=scores,
            episodes=episodes,
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
        for arm in ARMS:
            response_paths[f"{model}:{arm}"] = str(response_path(model, arm).resolve())

    if set(models) != set(MODEL_ORDER):
        raise RuntimeError(
            f"Model set mismatch: expected {sorted(MODEL_ORDER)}, found {sorted(models)}"
        )
    incomplete = {
        model: result["record_counts"]
        for model, result in models.items()
        if any(result["record_counts"][arm] != 1_250 for arm in ARMS)
    }
    if incomplete:
        raise RuntimeError(f"Incomplete response files: {incomplete}")

    result = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {model: models[model] for model in MODEL_ORDER},
        "not_for_paper": False,
        "primary_stratum": "k=0 (zero City C demonstrations)",
        "response_paths": response_paths,
        "source_interim": str(INTERIM_PATH.resolve()),
        "status": "complete_posthoc_comparison",
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paper_result = {
        key: value
        for key, value in result.items()
        if key not in {"response_paths", "source_interim"}
    }
    PAPER_OUTPUT_PATH.write_text(
        json.dumps(paper_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Coin City symbol-context comparison — final results",
        "",
        "Status: complete post-freeze exploratory comparison. Included in the paper.",
        "",
        "Primary stratum: k=0 (no City C demonstrations), with 250 frozen episode "
        "tasks per arm and matched complete-case n shown below. "
        f"Bootstrap draws: {BOOTSTRAP_DRAWS:,}.",
        "",
        "| Model | n | No-context MAE | Semantic MAE | Symbol MAE | Symbol − no-context ΔMAE [95% CI] | No-context ρ | Symbol ρ | Symbol − no-context Δρ [95% CI] | No-context acc. | Symbol acc. | Δacc. [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(primary_row(model, models[model]) for model in MODEL_ORDER)
    lines.extend(
        (
            "",
            "## Within-family scaling curves across demonstration counts",
            "",
            "| Model | k | n | Symbol − no-context ΔMAE [95% CI] | Δρ [95% CI] | Δacc. [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for model in (
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
    ):
        lines.extend(scaling_curve_row(model, k, models[model]) for k in range(5))

    def k0(model: str) -> dict:
        return models[model]["curves"]["0"]

    def delta_rho(model: str) -> str:
        return interval(
            k0(model)["paired_discrimination"]["symbol_minus_no_context_rho"],
            "difference",
        )

    def delta_mae(model: str) -> str:
        return interval(
            k0(model)["paired_forecast_mae"]["symbol_minus_no_context"],
            "mean",
        )

    qwen3_rho = ", ".join(
        delta_rho(model)
        for model in ("Qwen3-4B-Instruct-2507", "Qwen3-8B", "Qwen3-14B")
    )
    llama_rho = ", ".join(
        delta_rho(model)
        for model in ("Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct")
    )
    qwen25_rho = ", ".join(
        delta_rho(model)
        for model in (
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen2.5-72B-Instruct",
        )
    )
    qwen25_mae = ", ".join(
        delta_mae(model)
        for model in (
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen2.5-72B-Instruct",
        )
    )
    frontier_rho = ", ".join(
        delta_rho(model)
        for model in ("claude-opus-4-8", "claude-opus-5", "gpt-5.6-sol")
    )
    frontier_mae = ", ".join(
        delta_mae(model)
        for model in ("claude-opus-4-8", "claude-opus-5", "gpt-5.6-sol")
    )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "Within-family scaling appears in both Qwen3 and Llama 3.1. "
            f"Qwen3 4B/8B/14B symbol-minus-no-context Δρ values are {qwen3_rho}; "
            f"Llama 3.1 8B/70B values are {llama_rho}. The larger checkpoint "
            "recovers the episode-local mapping in both families.",
            "",
            "The boundary is family-specific rather than a universal parameter "
            f"cutoff: Qwen2.5 7B/32B/72B Δρ values are {qwen25_rho}. Their "
            f"corresponding ΔMAE values are {qwen25_mae}, so mapping recovery "
            "does not by itself guarantee calibrated numerical use.",
            "",
            "At frontier scale, both capabilities coincide. Claude Opus 4.8, "
            f"Claude Opus 5, and GPT-5.6 Sol have Δρ values {frontier_rho} and "
            f"ΔMAE values {frontier_mae}. DeepSeek V4 Pro has Δρ "
            f"{delta_rho('DeepSeek-V4-Pro')} and ΔMAE "
            f"{delta_mae('DeepSeek-V4-Pro')}; Kimi K3 has Δρ "
            f"{delta_rho('FW-Kimi-K3')} and ΔMAE {delta_mae('FW-Kimi-K3')}.",
            "",
            "For the large Qwen2.5 models and Llama 3.1-70B, the zero-shot "
            "discrimination advantage fades as City C observations accumulate. "
            "The induced label is most useful before direct target evidence is available.",
            "",
            "Reading guide: positive Δρ/Δaccuracy indicates better arbitrary-label discrimination; negative ΔMAE indicates better numerical forecasts. These properties can move in different directions.",
            "",
            "Original model responses are preserved. Arithmetic-only reparsing uses no additional model calls; omitted prediction fields remain unresolved.",
            "",
        )
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {PAPER_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
