#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


POLY = Path(__file__).resolve().parents[1]
SCRIPTS = POLY / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_hosted_scale_holdout as hosted  # noqa: E402
import evaluate_hosted_scale_holdout_liv as liv_adapter  # noqa: E402


def test_liv_adapter_changes_only_gpt_and_claude_transport() -> None:
    fake = SimpleNamespace(
        PROVIDERS={name: dict(config) for name, config in hosted.PROVIDERS.items()},
        __file__="base.py",
    )

    liv_adapter.configure(fake)

    assert fake.PROVIDERS["gpt-5.6-sol"]["endpoint"] == (
        "https://liv.services.ai.azure.com/openai/v1"
    )
    assert fake.PROVIDERS["claude-opus-4-8"]["endpoint"] == (
        "https://liv.services.ai.azure.com/anthropic"
    )
    assert fake.PROVIDERS["gpt-5.6-sol"]["key_env"] == "LIV_AZURE_API_KEY"
    assert fake.PROVIDERS["claude-opus-4-8"]["key_env"] == "LIV_AZURE_API_KEY"
    assert fake.PROVIDERS["FW-Kimi-K3"] == hosted.PROVIDERS["FW-Kimi-K3"]
    assert fake.PROVIDERS["DeepSeek-V4-Pro"] == hosted.PROVIDERS["DeepSeek-V4-Pro"]
    assert Path(fake.__file__).name == "evaluate_hosted_scale_holdout_liv.py"


def test_liv_adapter_rejects_unregistered_models() -> None:
    assert liv_adapter.requested_model(["--model", "gpt-5.6-sol"]) == ("gpt-5.6-sol")
    with pytest.raises(SystemExit, match="permits only"):
        liv_adapter.requested_model(["--model", "FW-Kimi-K3"])
