from __future__ import annotations

import json
import sys
from pathlib import Path


TRANSFER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRANSFER_ROOT))

import monitor_campaign  # noqa: E402


def test_locked_result_lines_reports_every_completed_model(tmp_path: Path) -> None:
    artifacts = {}
    for index, model in enumerate(("qwen3_8b", "llama3_1_8b")):
        path = tmp_path / f"{model}.summary.json"
        path.write_text(
            json.dumps(
                {
                    "model_key": model,
                    "models": {"base": {"brier": 0.2 + index, "parse_coverage": 1.0}},
                    "comparisons_brier": {
                        "trained_seed_mean_minus_base": {
                            "estimate": -0.1,
                            "ci95_low": -0.2,
                            "ci95_high": -0.05,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts[str(index)] = {"locked_summary": str(path)}

    rendered = "\n".join(monitor_campaign.locked_result_lines(artifacts))

    assert "Locked 3B result (qwen3_8b)" in rendered
    assert "Locked 3B result (llama3_1_8b)" in rendered
    assert rendered.count("trained_seed_mean_minus_base") == 2
