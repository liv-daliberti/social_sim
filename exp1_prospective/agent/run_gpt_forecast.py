#!/usr/bin/env python3
"""Stage 1 (GPT): Thin wrapper around run_forecast.py for gpt-5.4-pro.

Uses the shared Azure AI Foundry project (forecasting-agents-resource) with
the forecasting-agent, running gpt-5.4-pro with web search — directly
comparable to the Claude Opus 4.8 and DeepSeek R1 pipelines.

Usage (from exp1_prospective/):
    export AZURE_AI_API_KEY=<key>
    python agent/run_gpt_forecast.py
    python agent/run_gpt_forecast.py --n 5 --k 5
    python agent/run_gpt_forecast.py --verbose
    python agent/run_gpt_forecast.py --dry-run

All other flags (--input, --out, --delay, --k, etc.) are forwarded to run_forecast.py.
Falls back to CLAUDE_AZURE_API_KEY if AZURE_AI_API_KEY is not set
(same Azure project endpoint).
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

GPT_ENDPOINT = "https://liv-forecast.services.ai.azure.com/api/projects/proj-default"
GPT_AGENT    = "forecasting-agent"
GPT_VERSION  = "2"
GPT_MODEL    = "gpt-5.4"

def main():
    api_key = (os.environ.get("AZURE_AI_API_KEY")
               or os.environ.get("CLAUDE_AZURE_API_KEY", ""))

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"         not in flags: extra += ["--model",         GPT_MODEL]
    if "--endpoint"      not in flags: extra += ["--endpoint",      GPT_ENDPOINT]
    # Pass empty agent-name to bypass agent_reference (new endpoint has no named agent)
    if "--agent-name"    not in flags: extra += ["--agent-name",    ""]
    if "--api-key"       not in flags and api_key: extra += ["--api-key", api_key]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
