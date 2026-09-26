#!/usr/bin/env python3
"""Build and validate the no-LLM stable-response design preview."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship import (
    CASE_NOISE_SD,
    CITY_SLOPE_SD,
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    FUTURE_CALLS,
    HARD_CALL_CAP,
    MODELS,
    NEWS_MAGNITUDES,
    PROMPT_ARMS,
    QUERY_NEWS_VALUES,
    REFERENCE_CASES,
    SEED_BASE,
    SHARED_PROMPT_INTRO,
    STARTING_POLL_RANGE,
    STRONG_SLOPE_MEAN,
    TARGET_CONTEXT_STRONG,
    TARGET_CONTEXT_WEAK,
    WEAK_SLOPE_MEAN,
    empirical_estimates,
    fit_change_slope,
    make_episode,
    make_prompt,
    predict_from_slope,
    prompt_sections,
    prompt_sha256,
    shared_prompt_closing,
    target_background,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mae(predictions: list[float], truths: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truths))))


def _pair_contract(rows: list[dict], *, complete_pairs: int) -> bool:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["pair"])].append(row)
    for pair in range(1, complete_pairs + 1):
        members = grouped[pair]
        if len(members) != 2:
            return False
        if members[0]["starting_poll"] != members[1]["starting_poll"]:
            return False
        if int(members[0]["net_news"]) != -int(members[1]["net_news"]):
            return False
    return True


def _matching_reference(episode: dict) -> list[dict]:
    if episode["target_strong"]:
        city = episode["strong_reference_city"]
    else:
        city = "B" if episode["strong_reference_city"] == "A" else "A"
    return episode["reference_a" if city == "A" else "reference_b"]


def main() -> None:
    if FUTURE_CALLS != 2_250 or FUTURE_CALLS > HARD_CALL_CAP:
        raise SystemExit("future call-count invariant failed")
    if (RUN / "responses").exists() and any((RUN / "responses").glob("*.jsonl")):
        raise SystemExit("refusing to rebuild after model responses exist")
    DESIGN.mkdir(parents=True, exist_ok=True)

    episodes = [make_episode(index) for index in range(EPISODES)]
    keys: list[dict] = []
    tasks = {arm: [] for arm in PROMPT_ARMS}
    for episode in episodes:
        for c_cases in C_CASE_LEVELS:
            identifier = task_id(episode["episode"], c_cases)
            keys.append(
                {
                    "task_id": identifier,
                    "episode": episode["episode"],
                    "c_cases": c_cases,
                    "gold_expected_poll": episode["gold_expected_poll"],
                    "baselines": empirical_estimates(episode, c_cases),
                    "target_strong": episode["target_strong"],
                    "cue_strong": episode["cue_strong"],
                    "cue_correct": episode["cue_correct"],
                    "strong_reference_city": episode["strong_reference_city"],
                }
            )
            for arm in PROMPT_ARMS:
                prompt = make_prompt(episode, c_cases, arm)
                validate_arm_prompt(prompt, arm)
                tasks[arm].append(
                    {
                        "task_id": identifier,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                    }
                )

    _write_jsonl(DESIGN / "episodes.jsonl", episodes)
    _write_jsonl(DESIGN / "answer_key.jsonl", keys)
    scoring_keys = [
        {
            field: row[field]
            for field in (
                "task_id",
                "episode",
                "c_cases",
                "gold_expected_poll",
                "baselines",
            )
        }
        for row in keys
    ]
    _write_jsonl(DESIGN / "scoring_key.jsonl", scoring_keys)
    for arm in PROMPT_ARMS:
        _write_jsonl(DESIGN / f"tasks_{arm}.jsonl", tasks[arm])

    example = episodes[0]
    example_c_cases = 4
    components = {
        "shared_intro": SHARED_PROMPT_INTRO,
        "sections": {
            arm: prompt_sections(example, example_c_cases, arm)
            for arm in PROMPT_ARMS
        },
        "shared_closing": shared_prompt_closing(example),
        "episode": example["episode"],
        "c_cases": example_c_cases,
    }
    (DESIGN / "example_prompt_components.json").write_text(
        json.dumps(components, indent=2, sort_keys=True) + "\n"
    )
    for arm in PROMPT_ARMS:
        (DESIGN / f"example_prompt_{arm}.txt").write_text(
            make_prompt(example, example_c_cases, arm) + "\n"
        )

    source_paths = {
        "engine": ROOT / "engine" / "coin_city_stable_relationship.py",
        "builder_validator": HERE / "build_coin_city_stable_relationship.py",
        "frozen_validator": HERE / "validate_coin_city_stable_relationship.py",
        "renderer": ROOT / "analysis" / "render_coin_city_stable_relationship.py",
        "runner": HERE / "run_coin_city_stable_relationship.py",
        "local_launcher": HERE / "run_coin_city_stable_relationship_local.sh",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("stable-response source missing: " + ", ".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "status": "frozen_design_preview_no_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": EPISODES,
        "city_c_case_levels": list(C_CASE_LEVELS),
        "reference_cases_per_city": REFERENCE_CASES,
        "arms": list(PROMPT_ARMS),
        "models_if_run_later": list(MODELS),
        "future_calls_if_run": FUTURE_CALLS,
        "planned_calls": FUTURE_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "seed_base": SEED_BASE,
        "context_alignment": "100_percent_correct_news_environment",
        "relationship_visibility": {
            "stable_within_city_sentence": True,
            "matched_equal_and_opposite_news_pairs": True,
            "estimator_or_formula_in_prompt": False,
        },
        "data_generating_process": {
            "starting_poll_range": list(STARTING_POLL_RANGE),
            "news_magnitudes": list(NEWS_MAGNITUDES),
            "query_news_values": list(QUERY_NEWS_VALUES),
            "strong_slope_mean": STRONG_SLOPE_MEAN,
            "weak_slope_mean": WEAK_SLOPE_MEAN,
            "city_slope_sd": CITY_SLOPE_SD,
            "case_noise_sd": CASE_NOISE_SD,
        },
        "displayed_analyst_benchmarks": {
            "city_c": "OLS through origin of poll change on net news, City C rows only",
            "abc": "OLS through origin of poll change on net news, all displayed A/B/C rows",
            "included_in_llm_prompt": False,
            "inputs": "displayed numeric rows and displayed query only",
        },
        "context_sentences": {
            "national_news": TARGET_CONTEXT_STRONG,
            "local_news": TARGET_CONTEXT_WEAK,
        },
        "task_sha256": {
            arm: _sha256(DESIGN / f"tasks_{arm}.jsonl") for arm in PROMPT_ARMS
        },
        "answer_key_sha256": _sha256(DESIGN / "answer_key.jsonl"),
        "scoring_key_sha256": _sha256(DESIGN / "scoring_key.jsonl"),
        "episodes_sha256": _sha256(DESIGN / "episodes.jsonl"),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    balance = Counter(
        (episode["strong_reference_city"], bool(episode["target_strong"]))
        for episode in episodes
    )
    prompt_rows = [row for arm in PROMPT_ARMS for row in tasks[arm]]
    curves = {"city_c": {}, "abc": {}, "structural_diagnostic": {}}
    for c_cases in C_CASE_LEVELS:
        level = [row for row in keys if row["c_cases"] == c_cases]
        truths = [row["gold_expected_poll"] for row in level]
        curves["city_c"][c_cases] = (
            None
            if c_cases == 0
            else _mae([row["baselines"]["baseline"] for row in level], truths)
        )
        curves["abc"][c_cases] = _mae(
            [row["baselines"]["abc_no_context"] for row in level], truths
        )
        structural = []
        for row in level:
            episode = episodes[row["episode"]]
            slope = fit_change_slope(
                _matching_reference(episode) + episode["target"][:c_cases]
            )
            structural.append(predict_from_slope(episode, slope))
        curves["structural_diagnostic"][c_cases] = _mae(structural, truths)

    c_curve = [curves["city_c"][n] for n in C_CASE_LEVELS if n > 0]
    abc_curve = [curves["abc"][n] for n in C_CASE_LEVELS]
    structural_curve = [
        curves["structural_diagnostic"][n] for n in C_CASE_LEVELS
    ]
    reference_separations = [
        abs(fit_change_slope(e["reference_a"]) - fit_change_slope(e["reference_b"]))
        for e in episodes
    ]
    checks = {
        "no_llm_responses": not any((RUN / "responses").glob("*.jsonl")),
        "exact_dimensions": (
            len(episodes) == EPISODES
            and all(len(tasks[arm]) == EPISODES * len(C_CASE_LEVELS) for arm in PROMPT_ARMS)
            and FUTURE_CALLS == 2_250 <= HARD_CALL_CAP
        ),
        "closest_50_episode_factorial_balance": balance == Counter(
            {("A", True): 13, ("A", False): 12, ("B", True): 12, ("B", False): 13}
        ),
        "context_always_correct": all(
            e["cue_correct"] and e["cue_strong"] == e["target_strong"] for e in episodes
        ),
        "three_reference_rows_per_city": all(
            len(e[name]) == REFERENCE_CASES
            for e in episodes
            for name in ("reference_a", "reference_b")
        ),
        "reference_rows_include_visible_matched_pair": all(
            _pair_contract(e[name], complete_pairs=1)
            for e in episodes
            for name in ("reference_a", "reference_b")
        ),
        "all_four_city_c_rows_form_two_visible_pairs": all(
            len(e["target"]) == 4 and _pair_contract(e["target"], complete_pairs=2)
            for e in episodes
        ),
        "prompts_state_stability_and_pairing": (
            "typical responsiveness to net news is stable" in SHARED_PROMPT_INTRO
            and "share a Pair label" in SHARED_PROMPT_INTRO
        ),
        "prompts_never_prescribe_an_estimator": all(
            all(term not in row["prompt"].lower() for term in ("regression", "coefficient", "weighted average", "calculate a slope"))
            for row in prompt_rows
        ),
        "prompt_hashes_and_contracts": all(
            prompt_sha256(row["prompt"]) == row["prompt_sha256"]
            for row in prompt_rows
        ),
        "reference_patterns_visibly_separated": float(np.median(reference_separations)) > 0.60,
        "city_c_benchmark_declines_every_case": all(
            later < earlier for earlier, later in zip(c_curve, c_curve[1:])
        ),
        "abc_benchmark_declines_every_case": all(
            later < earlier for earlier, later in zip(abc_curve, abc_curve[1:])
        ),
        "abc_has_clear_sparse_data_benefit": (
            c_curve[0] - abc_curve[1] > 1.0
            and c_curve[1] - abc_curve[2] > 0.7
        ),
        "benchmarks_converge_by_four": abs(c_curve[-1] - abc_curve[-1]) < 0.20,
        "structural_matching_diagnostic_beats_abc": all(
            structural_curve[n] < abc_curve[n] for n in C_CASE_LEVELS
        ),
        "analyst_benchmarks_use_displayed_rows_only": manifest["displayed_analyst_benchmarks"]["inputs"] == "displayed numeric rows and displayed query only",
        "source_hashes_frozen": all(
            _sha256(path) == manifest["source_sha256"][name]
            for name, path in source_paths.items()
        ),
    }
    validation = {
        "experiment": EXPERIMENT,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "benchmark_mae": {
            name: {str(k): values[k] for k in C_CASE_LEVELS}
            for name, values in curves.items()
        },
        "median_fitted_reference_slope_separation": float(np.median(reference_separations)),
        "final_absolute_benchmark_gap": abs(c_curve[-1] - abc_curve[-1]),
        "future_calls_if_run": FUTURE_CALLS,
        "llm_calls_made": 0,
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("stable-response preflight failed: " + ", ".join(failed))
    print(
        f"Built and validated {EXPERIMENT}: {len(checks)} checks passed; "
        "0 LLM calls made"
    )


if __name__ == "__main__":
    main()
