#!/usr/bin/env python3
"""Correct the registered-response scale in the v2 output tracking runner.

The registered score files' ``predicted_response`` field is already a forecast
contrast, on the same scale as ``truth_response``.  The first-pass analysis
treated it as a level and subtracted ``prior_response`` again.  This versioned
shim preserves the frozen parser/bootstrap code while adding the prior before
that legacy subtraction, so the value entering the analysis is exactly the
registered predicted response.  All-failed tasks retain the conservative
population-prior imputation defined in ``run_output_tracking_v2``.
"""

from __future__ import annotations

import numpy as np

import run_output_tracking_v2 as runner


_uncorrected_condition_analysis = runner._condition_analysis


def condition_analysis_on_registered_response_scale(
    *, metadata, output_means, tasks, k, repetitions
):
    task_by_id = {row["task_id"]: row for row in tasks}
    level_encoded = {
        task_id: np.asarray(value, dtype=float)
        + np.asarray(task_by_id[task_id]["prior_response"], dtype=float)
        for task_id, value in output_means.items()
    }
    return _uncorrected_condition_analysis(
        metadata=metadata,
        output_means=level_encoded,
        tasks=tasks,
        k=k,
        repetitions=repetitions,
    )


runner._condition_analysis = condition_analysis_on_registered_response_scale


if __name__ == "__main__":
    runner.analysis.main()
