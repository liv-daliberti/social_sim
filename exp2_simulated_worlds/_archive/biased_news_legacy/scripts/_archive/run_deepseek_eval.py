#!/usr/bin/env python3
"""Biased-news eval using DeepSeek-V4-Pro via Azure AI Foundry.

Directly comparable to the Claude and GPT eval pipelines.

Usage (from biased_news/):
    export DEEPSEEK_AZURE_API_KEY=<key>
    python scripts/run_deepseek_eval.py dryrun --max-tasks 4
    python scripts/run_deepseek_eval.py eval
    python scripts/run_deepseek_eval.py eval --max-tasks 100

All other flags are forwarded to run_grpo.py.
Falls back to CLAUDE_AZURE_API_KEY if DEEPSEEK_AZURE_API_KEY is not set.
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEEPSEEK_ENDPOINT = (
    "https://forecasting-agents-resource.services.ai.azure.com"
    "/api/projects/forecasting-agents/openai/v1"
)
DEEPSEEK_MODEL = "DeepSeek-V4-Pro"
DEFAULT_OUT    = "reports/eval_deepseek"


def main() -> None:
    api_key = (
        os.environ.get("DEEPSEEK_AZURE_API_KEY")
        or os.environ.get("CLAUDE_AZURE_API_KEY", "")
    )

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra: list[str] = []
    if "--model"    not in flags: extra += ["--model",    DEEPSEEK_MODEL]
    if "--api-base" not in flags: extra += ["--api-base", DEEPSEEK_ENDPOINT]
    if "--api-key"  not in flags and api_key: extra += ["--api-key", api_key]
    if "--azure-auth" not in flags: extra += ["--azure-auth"]
    if "--output-dir" not in flags: extra += ["--output-dir", DEFAULT_OUT]

    if not api_key and "--api-key" not in flags:
        print("Set DEEPSEEK_AZURE_API_KEY (or CLAUDE_AZURE_API_KEY) or pass --api-key")
        sys.exit(1)

    cmd = [sys.executable, str(_HERE / "run_grpo.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
