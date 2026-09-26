#!/usr/bin/env python3
"""Run one FW-Kimi-K3 arm on the frozen Coin City n=250 design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_coin_city_stable_relationship_deepseek_v4_pro as implementation


MODEL = "FW-Kimi-K3"
KEY_ENV = "KIMI_K3_AZURE_API_KEY"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_384
PILOT_ARCHIVES = (
    implementation.RUN
    / "pilots"
    / "fw_kimi_k3_default_reasoning_max4096_20260810",
    implementation.RUN
    / "pilots"
    / "fw_kimi_k3_low_reasoning_max4096_20260810",
    implementation.RUN
    / "pilots"
    / "fw_kimi_k3_low_reasoning_max16384_hard_task_test_20260810",
)

implementation.MODEL = MODEL
implementation.MODEL_LABEL = "Kimi K3"
implementation.KEY_ENV = KEY_ENV
implementation.base.MODEL_CONFIG[MODEL] = {
    "endpoint": implementation.ENDPOINT,
    "keys": (KEY_ENV,),
    "claude": False,
}


def _kimi_chat_completions_call(
    client,
    model,
    messages,
    max_tokens,
    temperature,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max(max_tokens, MAX_OUTPUT_TOKENS),
        temperature=temperature,
        reasoning_effort=REASONING_EFFORT,
    )


implementation.base.chat_call = _kimi_chat_completions_call


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
                "reasoning_effort": REASONING_EFFORT,
                "effective_max_output_tokens": MAX_OUTPUT_TOKENS,
                "excluded_configuration_pilots": [
                    str(path) for path in PILOT_ARCHIVES
                ],
                "excluded_archived_pilot_attempts": 118,
                "excluded_archived_pilot_parsed": 84,
                "excluded_archived_pilot_unparseable": 34,
                "excluded_task_shaped_configuration_calls": 119,
                "configuration_audit": str(
                    implementation.RUN
                    / "pilots"
                    / "fw_kimi_k3_configuration_audit.json"
                ),
                "clean_run_after_excluded_pilot": True,
            }
        )
        manifest_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
