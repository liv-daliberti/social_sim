#!/usr/bin/env python3
"""Run the harder Coin City population through the frozen open-model runner."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_coin_city_symbol_context_open_model as base  # noqa: E402
from engine.coin_city_robustness_population import EXPERIMENT  # noqa: E402

RUN = ROOT / "data" / EXPERIMENT
base.EXPERIMENT = EXPERIMENT
base.RUN = RUN
base.DESIGN = RUN / "design"
base.DEFAULT_OUTDIR = RUN / "responses" / "unspecified_model"
base.EXPECTED_TASKS = 1_250

if __name__ == "__main__":
    base.main()
