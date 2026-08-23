#!/usr/bin/env python3
"""Independently audit the prompt-only empirical coin-city benchmarks."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_expanded_aligned_v1"
DESIGN = RUN / "design"
LIVE = RUN / "live"

CITY_BLOCK = re.compile(
    r"CITY ([ABC])\n.*?\n(\| Case .*?)(?=\n\nCITY |\n\nA new City C)",
    re.DOTALL,
)
ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*50\.0\s*\|\s*\+8\s*\|\s*([0-9.]+)\s*\|$",
    re.MULTILINE,
)


def read_jsonl(path: Path, *, tolerate_partial_tail: bool = False) -> list[dict]:
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_partial_tail and index == len(lines) - 1:
                continue
            raise
    return rows


def visible_rows(prompt: str) -> dict[str, list[float]]:
    result = {
        match.group(1): [float(value) for value in ROW.findall(match.group(2))]
        for match in CITY_BLOCK.finditer(prompt)
    }
    if not result or any(not values for values in result.values()):
        raise AssertionError("prompt table parsing failed")
    return result


def empirical_abc(rows: dict[str, list[float]]) -> dict[str, float]:
    a, b, c = rows["A"], rows["B"], rows["C"]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    grand = 0.5 * (mean_a + mean_b)
    within_ss = sum((x - mean_a) ** 2 for x in a) + sum(
        (x - mean_b) ** 2 for x in b
    )
    within = within_ss / (len(a) + len(b) - 2)
    reference_dispersion = (mean_a - grand) ** 2 + (mean_b - grand) ** 2
    mean_sampling = 0.5 * within * (1.0 / len(a) + 1.0 / len(b))
    between = max(0.0, reference_dispersion - mean_sampling)
    c_mean = sum(c) / len(c)
    c_sampling = within / len(c)
    weight = between / (between + c_sampling) if between + c_sampling > 0 else 1.0
    return {
        "estimate": weight * c_mean + (1.0 - weight) * grand,
        "reference_grand_mean": grand,
        "within_variance_estimate": within,
        "between_variance_estimate": between,
        "city_c_weight": weight,
    }


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-11)


def main() -> None:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    baseline = {
        row["task_id"]: row
        for row in read_jsonl(DESIGN / "tasks_baseline.jsonl")
    }
    abc = {
        row["task_id"]: row
        for row in read_jsonl(DESIGN / "tasks_abc_no_context.jsonl")
    }
    records = {
        row["task_id"]: row
        for row in read_jsonl(LIVE / "empirical_benchmarks.jsonl")
    }
    assert len(baseline) == len(abc) == len(records) == 500
    assert set(baseline) == set(abc) == set(records)

    expected_cases = {int(k): int(v) for k, v in manifest["cases_by_round"].items()}
    prompts_by_episode: dict[int, dict[int, dict[str, list[float]]]] = {}
    for task_id, record in records.items():
        c_only = visible_rows(baseline[task_id]["prompt"])
        all_cities = visible_rows(abc[task_id]["prompt"])
        assert set(c_only) == {"C"}
        assert set(all_cities) == {"A", "B", "C"}
        assert len(all_cities["A"]) == len(all_cities["B"]) == 6
        k = int(record["k"])
        assert len(c_only["C"]) == len(all_cities["C"]) == expected_cases[k]
        assert c_only["C"] == all_cities["C"]
        c_estimate = sum(c_only["C"]) / len(c_only["C"])
        assert close(c_estimate, record["city_c_regression"]["estimate"])
        recomputed = empirical_abc(all_cities)
        stored = record["abc_no_context_empirical_partial_pool"]
        for field, value in recomputed.items():
            assert close(value, stored[field]), (task_id, field)
        prompts_by_episode.setdefault(int(record["episode"]), {})[k] = all_cities

    prefix_checks = 0
    for rounds in prompts_by_episode.values():
        fullest = rounds[max(rounds)]["C"]
        reference_a = rounds[min(rounds)]["A"]
        reference_b = rounds[min(rounds)]["B"]
        for rows in rounds.values():
            assert rows["C"] == fullest[: len(rows["C"])]
            assert rows["A"] == reference_a and rows["B"] == reference_b
            prefix_checks += 1
    assert prefix_checks == 500

    estimator_source = (ROOT / "analysis" / "empirical_coin_city_estimators.py").read_text()
    assert "from engine" not in estimator_source
    assert "CITY_ENDPOINT_SD" not in estimator_source
    assert "CASE_NOISE_SD" not in estimator_source

    progress = json.loads((LIVE / "empirical_progress.json").read_text())
    assert progress["analysis_revision"] == "prompt_only_empirical_v1"
    assert progress["scoring_only_field"] == "gold_expected_poll"
    assert set(progress["discarded_before_scoring"]) == {
        "original simulator baselines",
        "target_high",
        "high_reference_city",
        "cue_high",
        "cue_correct",
    }

    latest = {}
    for path in (RUN / "responses").glob("responses_*.jsonl"):
        for row in read_jsonl(path, tolerate_partial_tail=True):
            latest[(row["model"], row["arm"], row["task_id"])] = row
    current_common = {}
    for k in manifest["rounds"]:
        cells = []
        for model in manifest["models"]:
            for arm in manifest["arms"]:
                cells.append(
                    {
                        task_id
                        for task_id, record in records.items()
                        if int(record["k"]) == int(k)
                        and latest.get((model, arm, task_id), {}).get("predicted_poll")
                        is not None
                    }
                )
        current_common[str(k)] = len(set.intersection(*cells))
        assert progress["curves"][str(k)]["common_n"] <= current_common[str(k)]

    result = {
        "status": "pass",
        "benchmarks_checked": len(records),
        "prompt_prefix_checks": prefix_checks,
        "simulation_constants_used_by_estimators": False,
        "latent_design_fields_passed_to_scoring": False,
        "rendered_common_n": {
            str(k): progress["curves"][str(k)]["common_n"]
            for k in manifest["rounds"]
        },
        "current_common_n_at_audit": current_common,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
