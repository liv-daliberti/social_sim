#!/usr/bin/env python3
"""Stage 3 (DeepSeek): Thin wrapper around run_updated_forecast.py for DeepSeek-V4-Pro.

Usage (from exp1_prospective/):
    export DEEPSEEK_AZURE_API_KEY=<key>
    python agent/deepseek_updated_forecast.py
    python agent/deepseek_updated_forecast.py --n 1 --k 2
    python agent/deepseek_updated_forecast.py --cf-direction pro_H1
    python agent/deepseek_updated_forecast.py --dry-run

All flags are forwarded to run_updated_forecast.py.
Falls back to CLAUDE_AZURE_API_KEY if DEEPSEEK_AZURE_API_KEY is not set.
"""

import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEEPSEEK_ENDPOINT = "https://cos-tiktok-annotation-a-resource.services.ai.azure.com"
DEEPSEEK_AGENT    = ""
DEEPSEEK_MODEL    = "DeepSeek-V4-Pro"

def main():
    api_key = os.environ.get("DEEPSEEK_AZURE_API_KEY") or os.environ.get("CLAUDE_AZURE_API_KEY", "")

    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra = []
    if "--model"      not in flags: extra += ["--model",      DEEPSEEK_MODEL]
    if "--endpoint"   not in flags: extra += ["--endpoint",   DEEPSEEK_ENDPOINT]
    if "--agent-name" not in flags: extra += ["--agent-name", DEEPSEEK_AGENT]
    if "--api-key"    not in flags and api_key: extra += ["--api-key", api_key]

    import subprocess
    cmd = [sys.executable, str(_HERE / "run_updated_forecast.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
