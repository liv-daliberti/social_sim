#!/usr/bin/env python3
"""Stage 1 (Claude): Thin wrapper around run_forecast.py for forecasting-agent-2.

forecasting-agent-2 runs claude-opus-4-8 with web search via Azure AI Foundry,
producing directly comparable forecasts to the GPT pipeline.

Usage (from exp1_prospective/):
    export CLAUDE_AZURE_API_KEY=<key>
    python agent/run_claude_forecast.py
    python agent/run_claude_forecast.py --n 5 --k 5
    python agent/run_claude_forecast.py --verbose
    python agent/run_claude_forecast.py --dry-run

All other flags (--input, --out, --delay, --k, etc.) are forwarded to run_forecast.py.
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CLAUDE_ENDPOINT = "https://forecasting-agents-resource.services.ai.azure.com/api/projects/forecasting-agents"
CLAUDE_AGENT    = "forecasting-agent-2"
CLAUDE_VERSION  = "2"
CLAUDE_MODEL    = "claude-opus-4-8"

def main():
    api_key = os.environ.get("CLAUDE_AZURE_API_KEY", "")

    # Build argv for run_forecast.py, injecting Claude-specific defaults
    # unless the caller already passed those flags explicitly
    argv = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"         not in flags: extra += ["--model",         CLAUDE_MODEL]
    if "--endpoint"      not in flags: extra += ["--endpoint",      CLAUDE_ENDPOINT]
    if "--agent-name"    not in flags: extra += ["--agent-name",    CLAUDE_AGENT]
    if "--agent-version" not in flags: extra += ["--agent-version", CLAUDE_VERSION]
    if "--api-key"       not in flags and api_key: extra += ["--api-key", api_key]
    # forecasting-agent-2 fails on the 4th turn (context depth limit); use 3-turn mode
    if "--no-third-turn" not in flags and "--third-turn" not in flags:
        extra += ["--no-third-turn"]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
