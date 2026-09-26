"""Compatibility name for the shared frozen-task shard runner.

The original Coin City wrapper is hash-locked in the frozen design manifest and
imports this historical module name.  Alias the module object itself so the
wrapper's configuration assignments apply directly to the maintained harness.
The superseded three-city v9 implementation lives under ``exp2_v2/_archive``.
"""

from __future__ import annotations

import sys

import run_frozen_task_shard as _implementation


_implementation.EXPERIMENT = "coin_city_stable_relationship_claude_n250_v4"
_implementation.PREFIX_LADDER = (0, 1, 2, 3, 4)
_implementation.MODEL_CONFIG.setdefault(
    "claude-opus-4-8",
    {
        "endpoint": "https://liv-forecast.services.ai.azure.com/anthropic",
        "keys": ("AZURE_AI_API_KEY", "CLAUDE_AZURE_API_KEY"),
        "claude": True,
    },
)
sys.modules[__name__] = _implementation
