#!/usr/bin/env python3
"""Validate FW-Kimi-K3 responses against the frozen Coin City task contract."""

from __future__ import annotations

import json

import validate_coin_city_stable_relationship_deepseek_v4_pro as implementation


MODEL = "FW-Kimi-K3"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_384

implementation.MODEL = MODEL


def main() -> None:
    implementation.main()
    checks = {}
    for arm in implementation.PROMPT_ARMS:
        path = (
            implementation.RESPONSES
            / f"responses_{MODEL}_{arm}.manifest.json"
        )
        manifest = json.loads(path.read_text())
        checks[arm] = {
            "reasoning_effort": (
                manifest.get("reasoning_effort") == REASONING_EFFORT
            ),
            "effective_max_output_tokens": (
                manifest.get("effective_max_output_tokens")
                == MAX_OUTPUT_TOKENS
            ),
            "zero_http_client_retries": (
                manifest.get("http_client_retries") == 0
            ),
            "clean_run_after_excluded_pilot": (
                manifest.get("clean_run_after_excluded_pilot") is True
            ),
            "configuration_audit_linked": (
                manifest.get("excluded_task_shaped_configuration_calls") == 119
                and bool(manifest.get("configuration_audit"))
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
        raise SystemExit("FW-Kimi-K3 configuration validation failed")


if __name__ == "__main__":
    main()
