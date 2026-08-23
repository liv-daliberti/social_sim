#!/usr/bin/env python3
"""Biased-news eval using gpt-5.4 via Azure AI Foundry.

Directly comparable to the Claude and DeepSeek eval pipelines.

Usage (from biased_news/):
    export AZURE_AI_API_KEY=<key>
    python scripts/run_gpt_eval.py dryrun --max-tasks 4
    python scripts/run_gpt_eval.py eval
    python scripts/run_gpt_eval.py eval --max-tasks 100

All other flags are forwarded to run_grpo.py.
Falls back to CLAUDE_AZURE_API_KEY if AZURE_AI_API_KEY is not set.
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

GPT_ENDPOINT = (
    "https://forecasting-agents-resource.services.ai.azure.com"
    "/api/projects/forecasting-agents/openai/v1"
)
GPT_MODEL   = "gpt-5.4"
DEFAULT_OUT = "reports/eval_gpt"


def main() -> None:
    api_key = (
        os.environ.get("AZURE_AI_API_KEY")
        or os.environ.get("CLAUDE_AZURE_API_KEY", "")
    )

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra: list[str] = []
    if "--model"    not in flags: extra += ["--model",    GPT_MODEL]
    if "--api-base" not in flags: extra += ["--api-base", GPT_ENDPOINT]
    if "--api-key"  not in flags and api_key: extra += ["--api-key", api_key]
    if "--azure-auth" not in flags: extra += ["--azure-auth"]
    if "--output-dir" not in flags: extra += ["--output-dir", DEFAULT_OUT]

    if not api_key and "--api-key" not in flags:
        print("Set AZURE_AI_API_KEY (or CLAUDE_AZURE_API_KEY) or pass --api-key")
        sys.exit(1)

    cmd = [sys.executable, str(_HERE / "run_grpo.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
