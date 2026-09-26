#!/usr/bin/env python3
"""Stage 3 (Claude): Thin wrapper around run_updated_forecast.py for claude-opus-4-8.

Usage (from exp1_prospective/):
    export CLAUDE_AZURE_API_KEY=<key>
    python agent/claude_updated_forecast.py
    python agent/claude_updated_forecast.py --n 1 --k 2
    python agent/claude_updated_forecast.py --cf-direction pro_H1
    python agent/claude_updated_forecast.py --dry-run

All flags are forwarded to run_updated_forecast.py.
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CLAUDE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/anthropic"
CLAUDE_MODEL    = "claude-opus-4-8"

def main():
    api_key = os.environ.get("AZURE_AI_API_KEY") or os.environ.get("CLAUDE_AZURE_API_KEY", "")

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"    not in flags: extra += ["--model",    CLAUDE_MODEL]
    if "--endpoint" not in flags: extra += ["--endpoint", CLAUDE_ENDPOINT]
    if "--api-key"  not in flags and api_key: extra += ["--api-key", api_key]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_updated_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
