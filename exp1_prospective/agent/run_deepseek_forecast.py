#!/usr/bin/env python3
"""Stage 1 (DeepSeek): Thin wrapper around run_forecast.py for deep-seek-forecast.

deep-seek-forecast runs DeepSeek R1 with web search via Azure AI Foundry,
producing directly comparable forecasts to the GPT and Claude pipelines.

Usage (from exp1_prospective/):
    export DEEPSEEK_AZURE_API_KEY=<key>
    python agent/run_deepseek_forecast.py
    python agent/run_deepseek_forecast.py --n 5 --k 5
    python agent/run_deepseek_forecast.py --verbose
    python agent/run_deepseek_forecast.py --dry-run

All other flags (--input, --out, --delay, --k, etc.) are forwarded to run_forecast.py.
Falls back to CLAUDE_AZURE_API_KEY if DEEPSEEK_AZURE_API_KEY is not set
(same Azure project endpoint).
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEEPSEEK_ENDPOINT = "https://forecasting-agents-resource.services.ai.azure.com/api/projects/forecasting-agents"
DEEPSEEK_AGENT    = "deep-seek-forecast"
DEEPSEEK_VERSION  = "2"
DEEPSEEK_MODEL    = "DeepSeek-V4-Pro"

def main():
    api_key = (os.environ.get("DEEPSEEK_AZURE_API_KEY")
               or os.environ.get("CLAUDE_AZURE_API_KEY", ""))

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"         not in flags: extra += ["--model",         DEEPSEEK_MODEL]
    if "--endpoint"      not in flags: extra += ["--endpoint",      DEEPSEEK_ENDPOINT]
    if "--agent-name"    not in flags: extra += ["--agent-name",    DEEPSEEK_AGENT]
    if "--agent-version" not in flags: extra += ["--agent-version", DEEPSEEK_VERSION]
    if "--api-key"       not in flags and api_key: extra += ["--api-key", api_key]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
