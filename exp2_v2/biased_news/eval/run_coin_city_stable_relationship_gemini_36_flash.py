#!/usr/bin/env python3
"""Run one Gemini 3.6 Flash arm on the frozen Coin City n=250 design.

This additive replication layer reuses the exact task files and prompt hashes
from the completed Claude Opus 4.8 run. Gemini is called through the Gemini API
OpenAI-compatible Chat Completions endpoint, so the request path matches the
DeepSeek and Kimi replications. Authentication uses an ephemeral project key
supplied in the environment; no key is stored in the repository.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import run_coin_city_stable_relationship_deepseek_v4_pro as implementation


MODEL = "gemini-3.6-flash"
MODEL_LABEL = "Gemini 3.6 Flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
KEY_ENV = "GEMINI_API_KEY"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_384
PILOT_ARCHIVES = ("pilots/gemini_36_flash_low_reasoning_max16384_20260811",)

# The harness records one terminal record per task, so a request the service
# rejects would destroy that task. Shards therefore share a file-locked schedule
# that spaces every request in the run evenly, and a rejected request is
# re-issued before the model ever sees the task. Retries here never re-ask a
# task the model actually answered: the single-model-attempt invariant is
# enforced one level up, in the harness.
#
# The interval is a guard rail rather than the binding constraint on this
# deployment, which accepted 40 simultaneous requests in 2.5s. Each rejection
# widens it, so the run settles at whatever pace the service accepts. (The
# Gemini 3.1 Pro preview deployment this replication first targeted is capped at
# 250 requests/day regardless of billing tier; see the pilots directory.)
MIN_REQUEST_INTERVAL_SECONDS = 0.15
MAX_REQUEST_INTERVAL_SECONDS = 8.0
MAX_SCHEDULE_BACKLOG_SECONDS = 120.0
REJECTION_INTERVAL_PENALTY_SECONDS = 0.1
# Widening must be reversible. A one-way penalty lets a handful of transient
# rejections throttle the whole run permanently: 26 of them once dragged this
# deployment from ~190 to 40 requests/minute and never recovered. Accepted
# requests therefore decay the interval back toward the floor.
ACCEPTED_INTERVAL_DECAY = 0.98
MAX_PRE_MODEL_RETRIES = 12
_RETRY_STATUS = (429, 500, 502, 503, 504)
_RETRY_DELAY = re.compile(r"[Pp]lease retry in ([0-9.]+)s")
_DAILY_QUOTA = "generate_requests_per_model_per_day"
_TRANSPORT_RETRIES = 0

implementation.MODEL = MODEL
implementation.MODEL_LABEL = MODEL_LABEL
implementation.ENDPOINT = ENDPOINT
implementation.PROJECT_ENDPOINT = ENDPOINT
implementation.KEY_ENV = KEY_ENV
implementation.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
implementation.base.MODEL_CONFIG[MODEL] = {
    "endpoint": ENDPOINT,
    "keys": (KEY_ENV,),
    "claude": False,
}


def _gemini_make_client(
    model: str,
    *,
    endpoint_override: Optional[str],
    timeout: float,
):
    if model != MODEL:
        raise RuntimeError(f"unsupported model: {model}")
    from openai import OpenAI

    key = os.environ.get(KEY_ENV, "")
    if not key:
        raise RuntimeError(f"no API key found in {KEY_ENV}")
    endpoint = endpoint_override or ENDPOINT
    implementation._AUTH_MODE = "project_api_key"
    client = OpenAI(
        base_url=endpoint,
        api_key=key,
        timeout=timeout,
        max_retries=0,
    )
    return client, endpoint


def _schedule_path() -> Path:
    path = implementation.RUN / "live" / "gemini_36_flash_request_schedule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _edit_schedule(claim: bool, penalty: float, decay: bool = False) -> float:
    """Read-modify-write the shared schedule under an exclusive lock."""
    with open(_schedule_path(), "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            state = json.loads(raw) if raw else {}
        except ValueError:
            state = {}
        interval = float(state.get("interval", MIN_REQUEST_INTERVAL_SECONDS))
        if penalty:
            interval = min(MAX_REQUEST_INTERVAL_SECONDS, interval + penalty)
        if decay:
            interval = max(
                MIN_REQUEST_INTERVAL_SECONDS, interval * ACCEPTED_INTERVAL_DECAY
            )
        now = time.time()
        next_start = float(state.get("next_start", 0.0))
        # A backlog longer than the queue of shards means the schedule has run
        # away from wall-clock time; collapse it rather than sleeping for hours.
        if next_start > now + MAX_SCHEDULE_BACKLOG_SECONDS:
            next_start = now
        start = max(now, next_start)
        state = {
            "interval": interval,
            "next_start": start + interval if claim else next_start,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state))
        fcntl.flock(handle, fcntl.LOCK_UN)
    return start if claim else interval


def _await_slot() -> None:
    """Block until this process's evenly spaced turn to issue a request."""
    delay = _edit_schedule(claim=True, penalty=0.0) - time.time()
    if delay > 0:
        time.sleep(delay)


def _widen_interval() -> float:
    """Slow every shard down after a rejection, without claiming a slot."""
    return _edit_schedule(claim=False, penalty=REJECTION_INTERVAL_PENALTY_SECONDS)


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    match = _RETRY_DELAY.search(str(exc))
    if match:
        return min(90.0, float(match.group(1)) + 1.0)
    return min(90.0, 5.0 * 2 ** (attempt - 1))


def _gemini_chat_completions_call(
    client,
    model,
    messages,
    max_tokens,
    temperature,
):
    global _TRANSPORT_RETRIES
    for attempt in range(1, MAX_PRE_MODEL_RETRIES + 2):
        _await_slot()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max(max_tokens, MAX_OUTPUT_TOKENS),
                temperature=temperature,
                reasoning_effort=REASONING_EFFORT,
            )
            _edit_schedule(claim=False, penalty=0.0, decay=True)
            return response
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            # A day-quota rejection will not clear within any sane retry budget.
            # Stop the shard so the remaining tasks stay pending instead of
            # burning retries and retiring as placeholder records.
            if _DAILY_QUOTA in str(exc):
                raise SystemExit(
                    "daily request quota exhausted for this deployment; "
                    "remaining tasks left pending for a later resume"
                )
            if status not in _RETRY_STATUS or attempt > MAX_PRE_MODEL_RETRIES:
                raise
            _TRANSPORT_RETRIES += 1
            delay = _retry_delay_seconds(exc, attempt)
            interval = _widen_interval()
            print(
                f"pre-model retry {attempt} after HTTP {status}; "
                f"sleeping {delay:.1f}s (shard retries={_TRANSPORT_RETRIES}, "
                f"interval now {interval:.2f}s)",
                flush=True,
            )
            time.sleep(delay + random.uniform(0.0, 2.0))
    raise RuntimeError("unreachable")


implementation.base._make_client = _gemini_make_client
implementation.base.chat_call = _gemini_chat_completions_call


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
                "api_protocol": "Gemini API OpenAI-compatible Chat Completions",
                "reasoning_effort": REASONING_EFFORT,
                "effective_max_output_tokens": MAX_OUTPUT_TOKENS,
                "http_client_retries": 0,
                "shared_request_schedule": {
                    "min_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
                    "max_interval_seconds": MAX_REQUEST_INTERVAL_SECONDS,
                    "rejection_penalty_seconds": REJECTION_INTERVAL_PENALTY_SECONDS,
                    "spacing": "even",
                },
                "pre_model_transport_retry_policy": {
                    "max_retries": MAX_PRE_MODEL_RETRIES,
                    "model_responses_never_repeated": True,
                    "retried_http_status": list(_RETRY_STATUS),
                    "shard_transport_retries": _TRANSPORT_RETRIES,
                },
                "excluded_configuration_pilots": [
                    str(implementation.RUN / path) for path in PILOT_ARCHIVES
                ],
                "excluded_task_shaped_configuration_calls": 15,
                "configuration_audit": str(
                    implementation.RUN / "pilots" / "gemini_36_flash_configuration_audit.json"
                ),
                "clean_run_after_excluded_pilot": True,
            }
        )
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
