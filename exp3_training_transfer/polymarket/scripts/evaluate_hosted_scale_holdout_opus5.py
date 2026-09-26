#!/usr/bin/env python3
"""Run the post-hoc Opus 5 reference through the verified ``liv`` transport.

This adapter adds only the deployment-specific settings established by the
earlier Coin City Opus 5 configuration audit. Prompts, task ordering, parsing,
five-call aggregation, fail-closed scoring, and clustered inference remain in
``evaluate_hosted_scale_holdout``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_hosted_scale_holdout as implementation  # noqa: E402


MODEL = "claude-opus-5"
LIV_ANTHROPIC_ENDPOINT = "https://liv.services.ai.azure.com/anthropic"
LIV_KEY_ENV = "LIV_AZURE_API_KEY"
DEFAULT_MAX_OUTPUT_TOKENS = 4_096


def make_opus5_client(config: dict[str, Any], timeout: float):
    """Use the project-key authentication verified for this deployment."""
    key = os.environ.get(config["key_env"], "")
    if not key:
        raise RuntimeError(f"no API key found in {config['key_env']}")
    from anthropic import AnthropicFoundry

    return AnthropicFoundry(
        azure_ad_token_provider=lambda: key,
        base_url=config["endpoint"],
        timeout=timeout,
        max_retries=0,
    )


def configure(module: ModuleType | Any = implementation) -> None:
    module.PROVIDERS[MODEL] = {
        "protocol": "anthropic_messages",
        "endpoint": LIV_ANTHROPIC_ENDPOINT,
        "key_env": LIV_KEY_ENV,
        "temperature": None,
        "top_p": None,
        "reasoning_effort": None,
        "default_max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }
    module.make_client = make_opus5_client
    # Bind manifests to this deployment adapter, not only the shared runner.
    module.__file__ = __file__


def requested_model(argv: list[str]) -> str:
    try:
        model = argv[argv.index("--model") + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit("the Opus 5 adapter requires --model") from exc
    if model != MODEL:
        raise SystemExit(f"the Opus 5 adapter permits only {MODEL}")
    return model


def main() -> None:
    requested_model(sys.argv[1:])
    configure()
    implementation.main()


if __name__ == "__main__":
    main()
