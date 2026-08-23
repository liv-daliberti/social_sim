#!/usr/bin/env python3
"""Run one frozen v12 model-by-arm shard."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import run_three_city_c2_v9_confirmatory as base
from build_three_city_c2_v12_tasks import validate_arm_prompt
from engine.three_city_c2_v12 import PROMPT_ARMS

_RUN = _ROOT / "data" / "three_city_c2_v12" / "full_k1_5_noisier_structural_choice"
_DESIGN = _RUN / "design"
_OUTDIR = _RUN / "responses"
base.PROMPT_ARMS = PROMPT_ARMS
base._RUN = _RUN
base._DESIGN = _DESIGN
base._OUTDIR = _OUTDIR
base._TASK_ID = re.compile(r"^c2v12_(\d{4})_k([1-5])$")
base.validate_arm_prompt = validate_arm_prompt


def main() -> None:
    if "--tasks" not in sys.argv and "--arm" in sys.argv:
        arm = sys.argv[sys.argv.index("--arm") + 1]
        sys.argv.extend(["--tasks", str(_DESIGN / f"tasks_c2_v12_{arm}.jsonl")])
    base.main()
    if "--dry-run" not in sys.argv and _OUTDIR.exists():
        for path in _OUTDIR.glob("responses_*.manifest.json"):
            record = json.loads(path.read_text())
            record["experiment"] = "three_city_c2_v12_noisier_structural_choice"
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
