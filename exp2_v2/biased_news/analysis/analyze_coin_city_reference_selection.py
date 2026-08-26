#!/usr/bin/env python3
"""Test whether Coin City forecasts track the cue-named reference city's own slope.

The correct-versus-no-context contrast cannot separate two accounts of the
semantic arm. Under a lexical-prior account the model maps the City C coverage
sentence straight to a remembered response magnitude and never reads City A or
City B's numbers. Under a reference-selection account the model uses the cue to
pick a reference city, estimates that city's slope from its four noisy cases, and
transfers the estimate. Both accounts predict the observed regime-level
correlation, because the cue and the latent regime coincide.

The two accounts separate once the regime is partialled out. Within a regime the
displayed reference rows still carry sampling noise: a through-origin fit to four
cases has a within-regime standard deviation of about .33 against a generator
standard deviation of .03. A lexical-prior forecaster is deterministic given the
cue, so its residual implied slope cannot track that noise. A reference-selection
forecaster inherits it.

For every deployment, arm and prefix this module therefore reports

    partial r( implied slope , reference OLS | regime named by the cue )

separately for the reference the City C label names and for the other reference.
Arms without a City C label are scored against the true regime and report the
regime-matched and regime-mismatched reference instead, which is the pooling
control: with no cue the two loadings should be equal.

No model call is made. Everything is recomputed from the frozen response files,
the frozen episode records and the frozen scoring key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
MATCHED_RESPONSES = RESPONSES / "symbol_control_matched_20260813"
OPEN_MODEL_RESPONSES = RESPONSES / "symbol_control_open_qwen3_4b_20260824"
COMPARISON = (
    ROOT / "local_results" / "symbol_context_model_comparison_20260824"
)
DEFAULT_OUTPUT = RUN / "analysis" / "reference_selection_results.json"
BOOTSTRAP_DRAWS = 2_000

PRIMARY_MODELS = (
    "claude-opus-4-8",
    "gpt-5.6-sol",
    "DeepSeek-V4-Pro",
    "FW-Kimi-K3",
    "gemini-3.6-flash",
    "claude-opus-5",
)
PRIMARY_ARMS = ("baseline", "abc_no_context", "abc_context", "abc_wrong_context")

# Arms whose City C sentence names one of the two reference regimes, and whether
# the named regime equals the episode's true target regime.
CUED_ARMS = {
    "abc_context": True,
    "abc_symbol_context": True,
    "abc_wrong_context": False,
}

SYMBOL_MODELS = (
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
SYMBOL_LOCAL_DIRS = {
    "Qwen3-4B-Instruct-2507": OPEN_MODEL_RESPONSES,
    "Qwen2.5-7B-Instruct": COMPARISON / "qwen2_5_7b",
    "Qwen3-8B": COMPARISON / "qwen3_8b",
    "Qwen2.5-14B-Instruct": COMPARISON / "qwen2_5_14b",
    "Qwen3-14B": COMPARISON / "qwen3_14b",
    "Qwen3-32B": COMPARISON / "qwen3_32b",
    "Qwen2.5-32B-Instruct": COMPARISON / "qwen2_5_32b",
    "Qwen2.5-72B-Instruct": COMPARISON / "qwen2_5_72b",
    "Llama-3.1-8B-Instruct": COMPARISON / "llama_3_1_8b",
    "Llama-3.1-70B-Instruct": COMPARISON / "llama_3_1_70b",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_for(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:8], 16)


def response_path(model: str, arm: str) -> Path:
    """Resolve the frozen response file, preferring a no-call reparse when present."""
    if arm == "abc_symbol_context":
        if model in SYMBOL_LOCAL_DIRS:
            directory = SYMBOL_LOCAL_DIRS[model]
            reparsed = directory.with_name(directory.name + "_reparsed")
            candidate = reparsed / f"responses_{model}_{arm}.jsonl"
            if candidate.exists():
                return candidate
            return directory / f"responses_{model}_{arm}.jsonl"
        hosted = COMPARISON / "hosted" / f"responses_{model}_{arm}.jsonl"
        if hosted.exists():
            return hosted
        return RESPONSES / f"responses_{model}_{arm}.jsonl"
    if model in SYMBOL_LOCAL_DIRS and model not in PRIMARY_MODELS:
        directory = SYMBOL_LOCAL_DIRS[model]
        reparsed = directory.with_name(directory.name + "_reparsed")
        candidate = reparsed / f"responses_{model}_{arm}.jsonl"
        if candidate.exists():
            return candidate
        return directory / f"responses_{model}_{arm}.jsonl"
    if model == "DeepSeek-V4-Pro" and arm in {"abc_no_context", "abc_context"}:
        matched = MATCHED_RESPONSES / f"responses_{model}_{arm}.jsonl"
        if matched.exists():
            return matched
    return RESPONSES / f"responses_{model}_{arm}.jsonl"


def latest_responses(model: str, arm: str) -> dict[str, dict]:
    path = response_path(model, arm)
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        old = latest.get(row["task_id"])
        if old is None or row.get("predicted_poll") is not None:
            latest[row["task_id"]] = row
    return latest


def through_origin_slope(rows: list[dict]) -> float:
    """Analyst fit of poll change on net news through the origin."""
    numerator = sum(
        row["net_news"] * (row["ending_poll"] - row["starting_poll"]) for row in rows
    )
    denominator = sum(row["net_news"] ** 2 for row in rows)
    return float(numerator / denominator) if denominator else 0.0


def residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), control])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_correlation(
    left: np.ndarray, right: np.ndarray, control: np.ndarray
) -> float | None:
    """Pearson correlation of left and right after removing the control."""
    if len(left) < 3:
        return None
    left_residual = residualize(left, control)
    right_residual = residualize(right, control)
    # A constant input leaves a residual that is zero only up to rounding, so an
    # exact test would let a meaningless correlation through.
    scale = max(np.std(left), np.std(right), 1.0)
    if min(np.std(left_residual), np.std(right_residual)) < 1e-9 * scale:
        return None
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def bootstrap_partial(
    left: np.ndarray,
    right: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict:
    observed = partial_correlation(left, right, control)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(left), size=len(left))
        value = partial_correlation(left[indices], right[indices], control[indices])
        if value is not None and np.isfinite(value):
            values.append(value)
    if observed is None or not values:
        return {"partial_r": None, "ci_95": [None, None]}
    low, high = np.quantile(values, [0.025, 0.975])
    return {"partial_r": observed, "ci_95": [float(low), float(high)]}


def episode_vectors(
    episodes: dict[int, dict],
    scores: dict[str, dict],
    task_ids: list[str],
    arm: str,
) -> dict[str, np.ndarray]:
    """Per-episode design quantities for one arm's complete cases."""
    named_regime, cue_reference, other_reference, true_regime, true_slope = (
        [],
        [],
        [],
        [],
        [],
    )
    starting_poll, net_news = [], []
    cue_is_truthful = CUED_ARMS.get(arm)
    for task_id in task_ids:
        episode = episodes[scores[task_id]["episode"]]
        strong = bool(episode["target_strong"])
        # The regime the City C sentence names; without a cue, the true regime.
        named = strong if cue_is_truthful is None else (strong == cue_is_truthful)
        a_slope = through_origin_slope(episode["reference_a"])
        b_slope = through_origin_slope(episode["reference_b"])
        a_is_strong = episode["strong_reference_city"] == "A"
        named_slope = a_slope if a_is_strong == named else b_slope
        other_slope = b_slope if a_is_strong == named else a_slope
        named_regime.append(float(named))
        cue_reference.append(named_slope)
        other_reference.append(other_slope)
        true_regime.append(float(strong))
        true_slope.append(float(episode["target_slope"]))
        starting_poll.append(float(episode["query_starting_poll"]))
        net_news.append(float(episode["query_net_news"]))
    return {
        "named_regime": np.asarray(named_regime),
        "cue_reference_ols": np.asarray(cue_reference),
        "other_reference_ols": np.asarray(other_reference),
        "true_regime": np.asarray(true_regime),
        "true_slope": np.asarray(true_slope),
        "query_starting_poll": np.asarray(starting_poll),
        "query_net_news": np.asarray(net_news),
    }


def analyze_cell(
    model: str,
    arm: str,
    prefix: int,
    *,
    responses: dict[str, dict],
    episodes: dict[int, dict],
    scores: dict[str, dict],
    draws: int,
) -> dict:
    task_ids = sorted(
        task_id
        for task_id, score in scores.items()
        if int(score["c_cases"]) == prefix
        and responses.get(task_id, {}).get("predicted_poll") is not None
    )
    if len(task_ids) < 3:
        return {"n": len(task_ids)}
    vectors = episode_vectors(episodes, scores, task_ids, arm)
    predictions = np.asarray(
        [float(responses[task_id]["predicted_poll"]) for task_id in task_ids]
    )
    implied = (predictions - vectors["query_starting_poll"]) / vectors["query_net_news"]
    control = vectors["named_regime"]
    cued = arm in CUED_ARMS
    return {
        "n": len(task_ids),
        "cue_names_a_reference": cued,
        "control": "regime named by the City C label" if cued else "true target regime",
        "cue_reference" if cued else "regime_matched_reference": bootstrap_partial(
            implied,
            vectors["cue_reference_ols"],
            control,
            seed=seed_for(model, arm, str(prefix), "cue"),
            draws=draws,
        ),
        "other_reference" if cued else "regime_mismatched_reference": bootstrap_partial(
            implied,
            vectors["other_reference_ols"],
            control,
            seed=seed_for(model, arm, str(prefix), "other"),
            draws=draws,
        ),
        "implied_slope_vs_named_regime_r": float(
            np.corrcoef(implied, control)[0, 1]
        )
        if np.std(control)
        else None,
    }


def reference_noise_floor(episodes: dict[int, dict]) -> dict:
    """How much of the reference slope is sampling noise rather than regime."""
    ordered = [episodes[key] for key in sorted(episodes)]
    regime = np.asarray([float(row["target_strong"]) for row in ordered])
    true_slope = np.asarray([float(row["target_slope"]) for row in ordered])
    matched = np.asarray(
        [
            through_origin_slope(
                row["reference_a"]
                if (row["strong_reference_city"] == "A") == bool(row["target_strong"])
                else row["reference_b"]
            )
            for row in ordered
        ]
    )
    return {
        "episodes": len(ordered),
        "regime_vs_true_slope_r": float(np.corrcoef(regime, true_slope)[0, 1]),
        "matched_reference_ols_vs_true_slope_r": float(
            np.corrcoef(matched, true_slope)[0, 1]
        ),
        "matched_reference_ols_within_regime_sd": float(
            np.std(residualize(matched, regime), ddof=1)
        ),
        "generator_within_regime_sd": 0.03,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    scores = {row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")}
    episodes = {row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")}

    primary: dict[str, dict] = {}
    for model in PRIMARY_MODELS:
        arms: dict[str, dict] = {}
        for arm in PRIMARY_ARMS:
            responses = latest_responses(model, arm)
            if not responses:
                continue
            arms[arm] = {
                str(prefix): analyze_cell(
                    model,
                    arm,
                    prefix,
                    responses=responses,
                    episodes=episodes,
                    scores=scores,
                    draws=args.bootstrap_draws,
                )
                for prefix in range(5)
            }
        primary[model] = {"arms": arms}

    symbol: dict[str, dict] = {}
    for model in SYMBOL_MODELS:
        arms = {}
        for arm in ("abc_no_context", "abc_symbol_context"):
            responses = latest_responses(model, arm)
            if not responses:
                continue
            arms[arm] = {
                "0": analyze_cell(
                    model,
                    arm,
                    0,
                    responses=responses,
                    episodes=episodes,
                    scores=scores,
                    draws=args.bootstrap_draws,
                )
            }
        if arms:
            symbol[model] = {"arms": arms}

    result = {
        "estimand": (
            "partial Pearson correlation between the forecast-implied City C slope "
            "and a reference city's through-origin OLS slope, controlling for the "
            "response regime named by the City C label"
        ),
        "bootstrap_draws": args.bootstrap_draws,
        "model_calls": 0,
        "reference_noise_floor": reference_noise_floor(episodes),
        "primary": primary,
        "symbol": symbol,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
