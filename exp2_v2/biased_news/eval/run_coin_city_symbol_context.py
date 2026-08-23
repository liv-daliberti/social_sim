#!/usr/bin/env python3
"""Run one model on the arbitrary-symbol control for Experiment 2.

This thin wrapper preserves the frozen additive-arm execution protocol and
changes only the selected prompt arm.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_coin_city_wrong_context as shared  # noqa: E402
from engine.coin_city_symbol_context_arm import (  # noqa: E402
    ARM,
    CONTEXT_ALIGNMENT,
    PROMPT_ARMS,
    validate_arm_prompt,
)


shared.ARM = ARM
shared.CONTEXT_ALIGNMENT = CONTEXT_ALIGNMENT
shared.PROMPT_ARMS = PROMPT_ARMS
shared.validate_arm_prompt = validate_arm_prompt
shared.base.PROMPT_ARMS = PROMPT_ARMS
shared.base.validate_arm_prompt = validate_arm_prompt


if __name__ == "__main__":
    shared.main()
