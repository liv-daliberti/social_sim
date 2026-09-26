#!/usr/bin/env python3
"""Run the registered GPT/Claude references through the ``liv`` resource.

This transport-only adapter leaves the frozen multi-provider evaluator unchanged
while selecting the resource that owns both deployments. Scientific settings,
prompts, parsing, scoring, and fail-closed behavior remain in the base evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_hosted_scale_holdout as implementation  # noqa: E402


ALLOWED_MODELS = frozenset({"claude-opus-4-8", "gpt-5.6-sol"})
LIV_OPENAI_ENDPOINT = "https://liv.services.ai.azure.com/openai/v1"
LIV_ANTHROPIC_ENDPOINT = "https://liv.services.ai.azure.com/anthropic"
LIV_KEY_ENV = "LIV_AZURE_API_KEY"


def configure(module: ModuleType | Any = implementation) -> None:
    """Apply only the pre-model resource and credential routing amendment."""
    module.PROVIDERS["gpt-5.6-sol"].update(
        endpoint=LIV_OPENAI_ENDPOINT,
        key_env=LIV_KEY_ENV,
    )
    module.PROVIDERS["claude-opus-4-8"].update(
        endpoint=LIV_ANTHROPIC_ENDPOINT,
        key_env=LIV_KEY_ENV,
    )
    # Manifests must identify this adapter, which is the executed analysis code.
    module.__file__ = __file__


def requested_model(argv: list[str]) -> str:
    try:
        model = argv[argv.index("--model") + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit("the liv adapter requires --model") from exc
    if model not in ALLOWED_MODELS:
        raise SystemExit("the liv adapter permits only claude-opus-4-8 or gpt-5.6-sol")
    return model


def main() -> None:
    requested_model(sys.argv[1:])
    configure()
    implementation.main()


if __name__ == "__main__":
    main()
