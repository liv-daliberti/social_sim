import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
sys.path.insert(0, str(EVAL))

import run_frozen_task_shard as runner
import run_three_city_c2_v9_confirmatory as compatibility


def test_compatibility_name_aliases_the_maintained_harness():
    assert compatibility is runner
    assert runner.EXPERIMENT == "coin_city_stable_relationship_claude_n250_v4"
    assert runner.PREFIX_LADDER == (0, 1, 2, 3, 4)


def test_reply_parser_uses_the_last_valid_prediction():
    parsed = runner._parse_reply(
        'scratch {"predicted_poll": -1} '
        'answer {"predicted_poll": 62.5, "rationale": "final"}'
    )
    assert parsed == {"predicted_poll": 62.5, "rationale": "final"}
    assert runner._parse_reply('{"predicted_poll": 101}')["predicted_poll"] is None


def test_response_state_ignores_corrupt_lines_and_tracks_terminal_attempts(tmp_path):
    path = tmp_path / "responses.jsonl"
    rows = [
        {"task_id": "parsed", "predicted_poll": 51.0},
        {
            "task_id": "answered_unparsed",
            "predicted_poll": None,
            "response_received": True,
        },
        {
            "task_id": "transport_failure",
            "predicted_poll": None,
            "response_received": False,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot-json\n")

    terminal, parsed = runner._response_state(path)

    assert terminal == {"parsed", "answered_unparsed"}
    assert parsed == {"parsed"}
