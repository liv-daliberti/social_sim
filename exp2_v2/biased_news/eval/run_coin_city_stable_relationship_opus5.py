#!/usr/bin/env python3
"""Run one Claude Opus 5 arm on the frozen Coin City n=250 design.

This additive replication layer reuses the exact task files and prompt hashes
from the completed Claude Opus 4.8 run. Opus 5 is called through the Foundry
Anthropic Messages API on the project's own endpoint.

Two deployment differences from the frozen Opus 4.8 runner are handled here
rather than in the shared harness, so the Opus 4.8 path stays byte-identical:

* Opus 5 rejects ``temperature`` as deprecated, so none is supplied. The shared
  Claude call already omitted it, and the Opus 4.8 run likewise supplied none.
* Opus 5 returns an extended-thinking block ahead of its answer, so the reply is
  read from the first ``text`` block rather than from ``content[0]``.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import run_coin_city_stable_relationship_deepseek_v4_pro as implementation


MODEL = "claude-opus-5"
MODEL_LABEL = "Claude Opus 5"
ENDPOINT = "https://liv.services.ai.azure.com/anthropic"
PROJECT_ENDPOINT = "https://liv.services.ai.azure.com/api/projects/proj-default"
KEY_ENV = "OPUS5_AZURE_API_KEY"
MAX_OUTPUT_TOKENS = 4_096
PILOT_ARCHIVES = ("pilots/opus5_thinking_max4096_20260811",)

# The harness records one terminal record per task, so a request the service
# rejects would retire that task without the model ever seeing it. Rejected
# requests are therefore re-issued before the model runs; this never re-asks a
# task the model actually answered, which is what the single-model-attempt
# invariant protects.
MAX_PRE_MODEL_RETRIES = 12
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)
_RETRY_AFTER = re.compile(r"retry[- ]after[\"']?[:=]\s*[\"']?([0-9.]+)", re.I)
_TRANSPORT_RETRIES = 0

implementation.MODEL = MODEL
implementation.MODEL_LABEL = MODEL_LABEL
implementation.ENDPOINT = ENDPOINT
implementation.PROJECT_ENDPOINT = PROJECT_ENDPOINT
implementation.KEY_ENV = KEY_ENV
implementation.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
implementation.base.MODEL_CONFIG[MODEL] = {
    "endpoint": ENDPOINT,
    "keys": (KEY_ENV,),
    "claude": True,
}


def _opus5_make_client(
    model: str,
    *,
    endpoint_override: Optional[str],
    timeout: float,
):
    if model != MODEL:
        raise RuntimeError(f"unsupported model: {model}")
    from anthropic import AnthropicFoundry

    key = os.environ.get(KEY_ENV, "")
    if not key:
        raise RuntimeError(f"no API key found in {KEY_ENV}")
    endpoint = endpoint_override or ENDPOINT
    implementation._AUTH_MODE = "project_api_key"
    client = AnthropicFoundry(
        azure_ad_token_provider=lambda: key,
        base_url=endpoint,
        timeout=timeout,
        max_retries=0,
    )
    return client, endpoint


def _answer_text(message) -> str:
    """Read the answer from the first text block, skipping thinking blocks."""
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text or ""
    return ""


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    match = _RETRY_AFTER.search(str(exc))
    if match:
        return min(120.0, float(match.group(1)) + 1.0)
    return min(120.0, 4.0 * 2 ** (attempt - 1))


def _opus5_messages_call(client, model, messages, max_tokens, temperature):
    """Anthropic Messages call shimmed into the harness's OpenAI reply shape.

    ``temperature`` is accepted and ignored: Opus 5 rejects the parameter, and
    the frozen design supplies none for Claude deployments.
    """
    global _TRANSPORT_RETRIES
    for attempt in range(1, MAX_PRE_MODEL_RETRIES + 2):
        try:
            message = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max(max_tokens, MAX_OUTPUT_TOKENS),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=_answer_text(message)),
                        finish_reason=getattr(message, "stop_reason", "stop"),
                    )
                ]
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status not in _RETRY_STATUS or attempt > MAX_PRE_MODEL_RETRIES:
                raise
            _TRANSPORT_RETRIES += 1
            delay = _retry_delay_seconds(exc, attempt)
            print(
                f"pre-model retry {attempt} after HTTP {status}; "
                f"sleeping {delay:.1f}s (shard retries={_TRANSPORT_RETRIES})",
                flush=True,
            )
            time.sleep(delay + random.uniform(0.0, 2.0))
    raise RuntimeError("unreachable")


implementation.base._make_client = _opus5_make_client
implementation.base.chat_call = _opus5_messages_call


def main() -> None:
    implementation.main()
    if "--dry-run" not in sys.argv:
        arm = sys.argv[sys.argv.index("--arm") + 1]
        outdir = implementation.OUTDIR
        if "--outdir" in sys.argv:
            outdir = Path(sys.argv[sys.argv.index("--outdir") + 1])
        manifest_path = (
            outdir
            / f"responses_{implementation.base._slug(MODEL)}_{arm}.manifest.json"
        )
        record = json.loads(manifest_path.read_text())
        record.update(
            {
                "replication_model": MODEL_LABEL,
                "deployment": MODEL,
                "api_protocol": "Foundry Anthropic Messages API",
                "extended_thinking": "default; answer read from the first text block",
                "temperature": None,
                "temperature_supplied": False,
                "effective_max_output_tokens": MAX_OUTPUT_TOKENS,
                "http_client_retries": 0,
                "pre_model_transport_retry_policy": {
                    "max_retries": MAX_PRE_MODEL_RETRIES,
                    "model_responses_never_repeated": True,
                    "retried_http_status": list(_RETRY_STATUS),
                    "shard_transport_retries": _TRANSPORT_RETRIES,
                },
                "excluded_configuration_pilots": [
                    str(implementation.RUN / path) for path in PILOT_ARCHIVES
                ],
                "excluded_task_shaped_configuration_calls": 17,
                "configuration_audit": str(
                    implementation.RUN / "pilots" / "opus5_configuration_audit.json"
                ),
                "clean_run_after_excluded_pilot": True,
            }
        )
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
