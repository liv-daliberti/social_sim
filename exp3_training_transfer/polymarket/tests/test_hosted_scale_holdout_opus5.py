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
import evaluate_hosted_scale_holdout_opus5 as opus5  # noqa: E402


def fake_module() -> SimpleNamespace:
    return SimpleNamespace(
        PROVIDERS={name: dict(config) for name, config in hosted.PROVIDERS.items()},
        make_client=hosted.make_client,
        __file__=hosted.__file__,
    )


def test_adapter_adds_only_verified_opus5_deployment() -> None:
    fake = fake_module()
    opus5.configure(fake)

    assert fake.PROVIDERS[opus5.MODEL] == {
        "protocol": "anthropic_messages",
        "endpoint": "https://liv.services.ai.azure.com/anthropic",
        "key_env": "LIV_AZURE_API_KEY",
        "temperature": None,
        "top_p": None,
        "reasoning_effort": None,
        "default_max_output_tokens": 4_096,
    }
    assert fake.make_client is opus5.make_opus5_client
    assert Path(fake.__file__).name == "evaluate_hosted_scale_holdout_opus5.py"


def test_opus5_client_uses_verified_project_key_auth(monkeypatch) -> None:
    captured = {}

    class FakeAnthropicFoundry:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AnthropicFoundry=FakeAnthropicFoundry),
    )
    monkeypatch.setenv("LIV_AZURE_API_KEY", "test-key")
    config = {
        "key_env": "LIV_AZURE_API_KEY",
        "endpoint": opus5.LIV_ANTHROPIC_ENDPOINT,
    }

    opus5.make_opus5_client(config, 30.0)

    assert "api_key" not in captured
    assert captured["azure_ad_token_provider"]() == "test-key"
    assert captured["base_url"] == opus5.LIV_ANTHROPIC_ENDPOINT
    assert captured["max_retries"] == 0


def test_opus5_extended_thinking_text_is_parsed() -> None:
    response = SimpleNamespace(
        id="msg_opus5",
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking="private reasoning"),
            SimpleNamespace(type="text", text='{"yes_prob": 0.64}'),
        ],
    )

    class Recorder:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return response

    recorder = Recorder()
    config = {
        "protocol": "anthropic_messages",
        "temperature": None,
        "top_p": None,
    }
    raw, _, _, controls = hosted.issue(
        SimpleNamespace(messages=recorder), opus5.MODEL, config, "prompt", 4_096
    )

    assert hosted.registered.parse_yes_probability(raw) == pytest.approx(0.64)
    assert recorder.kwargs["max_tokens"] == 4_096
    assert "temperature" not in recorder.kwargs
    assert controls == {"temperature": None, "top_p": None}


def test_adapter_rejects_other_models() -> None:
    with pytest.raises(SystemExit, match="permits only"):
        opus5.requested_model(["--model", "claude-opus-4-8"])
