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


class Recorder:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.result


def test_registered_hosted_inputs_and_reporting_boundary() -> None:
    rows, registration = hosted.validate_inputs("holdout", hosted.REGISTRATION)

    assert len(rows) == 318
    assert registration["classification"] == "post_hoc_descriptive_extension"
    assert set(hosted.PROVIDERS) == {
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "FW-Kimi-K3",
        "DeepSeek-V4-Pro",
    }
    assert registration["hosted_reporting"]["placement"] == (
        "separate_off_the_shelf_reference_panel"
    )
    assert registration["hosted_reporting"]["decoder_identical_claim"] is False
    assert hosted.PROVIDERS["gpt-5.6-sol"]["key_env"] == "DEEPSEEK_AZURE_API_KEY"
    assert (
        hosted.PROVIDERS["DeepSeek-V4-Pro"]["endpoint"]
        == "https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1"
    )


def test_foundry_clients_use_resource_api_key_headers(monkeypatch) -> None:
    anthropic_kwargs = {}
    openai_kwargs = {}

    class FakeAnthropicFoundry:
        def __init__(self, **kwargs):
            anthropic_kwargs.update(kwargs)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            openai_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AnthropicFoundry=FakeAnthropicFoundry),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("CLAUDE_AZURE_API_KEY", "claude-test-key")
    monkeypatch.setenv("DEEPSEEK_AZURE_API_KEY", "openai-test-key")

    hosted.make_client(hosted.PROVIDERS["claude-opus-4-8"], 30.0)
    hosted.make_client(hosted.PROVIDERS["gpt-5.6-sol"], 30.0)

    assert anthropic_kwargs["api_key"] == "claude-test-key"
    assert "azure_ad_token_provider" not in anthropic_kwargs
    assert openai_kwargs["api_key"] == "placeholder"
    assert openai_kwargs["default_headers"] == {"api-key": "openai-test-key"}


def test_anthropic_transport_records_provider_native_sampling() -> None:
    response = SimpleNamespace(
        id="msg_1",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text='{"yes_prob": 0.61}')],
    )
    recorder = Recorder(response)
    client = SimpleNamespace(messages=recorder)
    config = hosted.PROVIDERS["claude-opus-4-8"]

    raw, response_id, finish, controls = hosted.issue(
        client, "claude-opus-4-8", config, "prompt", 128
    )

    assert hosted.registered.parse_yes_probability(raw) == pytest.approx(0.61)
    assert response_id == "msg_1"
    assert finish == "end_turn"
    assert "temperature" not in recorder.kwargs
    assert "top_p" not in recorder.kwargs
    assert recorder.kwargs["max_tokens"] == 128
    assert controls == {"temperature": None, "top_p": None}


def test_gpt_reasoning_transport_discloses_native_sampling() -> None:
    response = SimpleNamespace(
        id="resp_1", status="completed", output_text='{"yes_prob": 0.42}'
    )
    recorder = Recorder(response)
    client = SimpleNamespace(responses=recorder)
    config = hosted.PROVIDERS["gpt-5.6-sol"]

    raw, response_id, finish, controls = hosted.issue(
        client, "gpt-5.6-sol", config, "prompt", 128
    )

    assert hosted.registered.parse_yes_probability(raw) == pytest.approx(0.42)
    assert response_id == "resp_1"
    assert finish == "completed"
    assert "temperature" not in recorder.kwargs
    assert "top_p" not in recorder.kwargs
    assert recorder.kwargs["reasoning"] == {"effort": "low"}
    assert controls["temperature"] is None
    assert controls["top_p"] is None


def test_openai_compatible_transport_uses_common_controls() -> None:
    choice = SimpleNamespace(
        message=SimpleNamespace(content='{"yes_prob": 0.73}'),
        finish_reason="stop",
    )
    response = SimpleNamespace(id="chat_1", choices=[choice])
    recorder = Recorder(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    config = hosted.PROVIDERS["FW-Kimi-K3"]

    raw, response_id, finish, controls = hosted.issue(
        client, "FW-Kimi-K3", config, "prompt", 128
    )

    assert hosted.registered.parse_yes_probability(raw) == pytest.approx(0.73)
    assert response_id == "chat_1"
    assert finish == "stop"
    assert recorder.kwargs["temperature"] == 0.7
    assert recorder.kwargs["top_p"] == 0.8
    assert recorder.kwargs["reasoning_effort"] == "low"
    assert controls["top_p"] == 0.8


def test_local_extension_uses_stochastic_development_and_exact_parser() -> None:
    wrapper = (SCRIPTS / "polymarket_rl_architecture_extension.sh").read_text()

    assert "--eval_temperature 0.7" in wrapper
    assert "--eval_top_p 0.8" in wrapper
    assert "--eval_top_k 20" in wrapper
    assert "--eval_n 5" in wrapper
    assert "--eval_temperature 0 " not in wrapper
    assert "from forecast_scoring import parse_yes_probability" in wrapper
    assert "probability = parse_yes_probability(output)" in wrapper
