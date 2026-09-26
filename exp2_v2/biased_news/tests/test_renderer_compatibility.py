from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_live_renderer_keeps_its_compatibility_base() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "analysis/render_coin_city_stable_relationship_claude_n250.py",
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--watch" in result.stdout
