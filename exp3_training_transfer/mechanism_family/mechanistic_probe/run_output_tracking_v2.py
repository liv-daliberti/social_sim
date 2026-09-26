#!/usr/bin/env python3
"""Run output tracking with a conservative registered parse-failure policy.

An all-failed stochastic task has no response vector to correlate.  Rather than
drop that episode, this runner imputes the frozen population-prior response,
which contributes zero episode-specific information, and records the count.
"""

from __future__ import annotations

import numpy as np

import analyze_output_tracking as analysis


_condition_analysis = analysis.condition_analysis


def condition_analysis_with_parse_policy(
    *, metadata, output_means, tasks, k, repetitions
):
    disclosure = metadata["disclosure"]
    relevant = [
        row for row in tasks
        if row["disclosure"] == disclosure and int(row["k"]) == int(k)
    ]
    missing = [row for row in relevant if row["task_id"] not in output_means]
    augmented = dict(output_means)
    for row in missing:
        augmented[row["task_id"]] = np.asarray(row["prior_response"], dtype=float)
    result, rows = _condition_analysis(
        metadata=metadata,
        output_means=augmented,
        tasks=tasks,
        k=k,
        repetitions=repetitions,
    )
    result["parse_failure_policy"] = (
        "all-failed task imputed with frozen population-prior response; "
        "partial failures averaged over parsed draws"
    )
    result["all_failed_tasks_imputed"] = len(missing)
    result["all_failed_development_tasks_imputed"] = sum(
        row["split"] == "dev" for row in missing
    )
    result["all_failed_test_tasks_imputed"] = sum(
        row["split"] == "test" for row in missing
    )
    return result, rows


analysis.condition_analysis = condition_analysis_with_parse_policy


if __name__ == "__main__":
    analysis.main()
