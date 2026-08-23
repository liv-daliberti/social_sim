#!/usr/bin/env python3
"""Runtime-only compatibility wrapper for the frozen v13 live display.

The v13 answer key renamed the neutral benchmark from ``abc_shrinkage`` to
``abc_no_structure``.  The inherited scorer still reads the former name.
This wrapper aliases that field in memory; it does not modify frozen tasks,
answer keys, responses, or manifest-tracked experiment source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from analysis import live_exact_structure_three_city_c2_v13 as live


_score_rows_frozen = live.analysis._score_rows


def _score_rows_compatible(responses, keys):
    compatible_keys = {}
    for task_id, key in keys.items():
        baselines = key.get("baselines", {})
        if "abc_shrinkage" in baselines or "abc_no_structure" not in baselines:
            compatible_keys[task_id] = key
            continue
        compatible_key = dict(key)
        compatible_baselines = dict(baselines)
        compatible_baselines["abc_shrinkage"] = baselines["abc_no_structure"]
        compatible_key["baselines"] = compatible_baselines
        compatible_keys[task_id] = compatible_key
    return _score_rows_frozen(responses, compatible_keys)


live.analysis._score_rows = _score_rows_compatible


if __name__ == "__main__":
    live.main()
