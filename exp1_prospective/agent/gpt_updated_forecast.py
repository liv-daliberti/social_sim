#!/usr/bin/env python3
"""Stage 3 (GPT): Thin wrapper around run_updated_forecast.py for gpt-5.4.

Usage (from exp1_prospective/):
    export AZURE_AI_API_KEY=<key>
    python agent/gpt_updated_forecast.py
    python agent/gpt_updated_forecast.py --n 1 --k 2
    python agent/gpt_updated_forecast.py --cf-direction pro_H1
    python agent/gpt_updated_forecast.py --dry-run

All flags are forwarded to run_updated_forecast.py.
Falls back to CLAUDE_AZURE_API_KEY if AZURE_AI_API_KEY is not set.
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

GPT_ENDPOINT = "https://liv-forecast.services.ai.azure.com"
GPT_AGENT    = "forecasting-agent"
GPT_MODEL    = "gpt-5.4"

def main():
    api_key = os.environ.get("AZURE_AI_API_KEY") or os.environ.get("CLAUDE_AZURE_API_KEY", "")

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"               not in flags: extra += ["--model",               GPT_MODEL]
    if "--endpoint"            not in flags: extra += ["--endpoint",            GPT_ENDPOINT]
    if "--agent-name"          not in flags: extra += ["--agent-name",          GPT_AGENT]
    if "--novelty-assessment"  not in flags: extra += ["--novelty-assessment"]
    if "--api-key"             not in flags and api_key: extra += ["--api-key", api_key]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_updated_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
