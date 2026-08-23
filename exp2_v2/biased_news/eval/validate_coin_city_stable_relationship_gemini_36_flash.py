#!/usr/bin/env python3
"""Validate Gemini 3.6 Flash responses against the frozen Coin City contract."""

from __future__ import annotations

import json

import validate_coin_city_stable_relationship_deepseek_v4_pro as implementation


MODEL = "gemini-3.6-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_384

implementation.MODEL = MODEL
implementation.ENDPOINT = ENDPOINT


def main() -> None:
    implementation.main()
    checks = {}
    for arm in implementation.PROMPT_ARMS:
        path = implementation.RESPONSES / f"responses_{MODEL}_{arm}.manifest.json"
        manifest = json.loads(path.read_text())
        checks[arm] = {
            "reasoning_effort": manifest.get("reasoning_effort") == REASONING_EFFORT,
            "effective_max_output_tokens": (
                manifest.get("effective_max_output_tokens") == MAX_OUTPUT_TOKENS
            ),
            "zero_http_client_retries": manifest.get("http_client_retries") == 0,
            "api_protocol": (
                manifest.get("api_protocol")
                == "Gemini API OpenAI-compatible Chat Completions"
            ),
            "configuration_audit_linked": (
                manifest.get("excluded_task_shaped_configuration_calls") == 15
                and bool(manifest.get("configuration_audit"))
            ),
            "shared_request_schedule_declared": (
                manifest.get("shared_request_schedule", {}).get("spacing") == "even"
                and manifest.get("shared_request_schedule", {}).get(
                    "min_interval_seconds", 0.0
                )
                > 0
            ),
            "model_responses_never_repeated": (
                manifest.get("pre_model_transport_retry_policy", {}).get(
                    "model_responses_never_repeated"
                )
                is True
            ),
        }
    passed = all(all(values.values()) for values in checks.values())
    print(
        json.dumps(
            {
                "model": MODEL,
                "configuration_checks": checks,
                "status": "passed" if passed else "failed",
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(f"{MODEL} configuration validation failed")


if __name__ == "__main__":
    main()
