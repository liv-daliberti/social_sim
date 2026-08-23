from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exp1_prospective.indirect_interventions.review_app.app import create_app


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CANDIDATES = ROOT / "data/development/pilot_candidates.jsonl"
REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
PACKET = ROOT / "data/development/pilot_v2_review_packet.json"
VALIDATOR = ROOT / "scripts/validate_annotations.py"
LABELS = {
    "increase_yes": "increase",
    "decrease_yes": "decrease",
    "no_material_effect": "no_effect",
}


def opaque_id(panel: str, candidate_id: str, role: str) -> str:
    raw = f"indirect-pilot-v2|{panel}|{candidate_id}|{role}".encode()
    return "review_" + hashlib.sha256(raw).hexdigest()[:18]


class ValidationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tempdir.name) / "complete.sqlite3"
        create_app({
            "TESTING": True,
            "SECRET_KEY": "validator-test",
            "DATABASE": self.db,
            "PACKET_PATH": PACKET,
            "REVIEWERS_PATH": REVIEWERS,
        })
        candidates = [
            json.loads(line) for line in CANDIDATES.read_text().splitlines() if line.strip()
        ]
        expected = {}
        for candidate in candidates:
            for role, context in candidate["contexts"].items():
                task_id = opaque_id("full_context", candidate["candidate_id"], role)
                expected[task_id] = LABELS[context["intended_label"]]

        reviewers = json.loads(REVIEWERS.read_text())["reviewers"]
        connection = sqlite3.connect(self.db)
        for reviewer in reviewers:
            for task_id in reviewer["task_ids"]:
                full = reviewer["panel"] == "full_context"
                values = (
                    reviewer["reviewer_id"],
                    reviewer["panel"],
                    task_id,
                    expected[task_id] if full else "ambiguous",
                    5,
                    "2026-08-13T00:00:00+00:00",
                    "2026-08-13T00:01:00+00:00",
                    60.0,
                )
                connection.execute(
                    """
                    INSERT INTO annotations (
                        reviewer_id, panel, task_id, direction, confidence,
                        started_at, submitted_at, duration_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def validate(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--db", str(self.db)],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_complete_clean_panel_passes(self) -> None:
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["pilot_ready_for_revision"])

    def test_missing_rating_fails_closed(self) -> None:
        connection = sqlite3.connect(self.db)
        connection.execute(
            "DELETE FROM annotations WHERE rowid = (SELECT MIN(rowid) FROM annotations)"
        )
        connection.commit()
        connection.close()
        result = self.validate()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stdout)["complete"])

    def test_shortcut_fixed_sign_leakage_fails(self) -> None:
        reviewers = json.loads(REVIEWERS.read_text())["reviewers"]
        leaked_task = next(
            reviewer["task_ids"][0]
            for reviewer in reviewers if reviewer["panel"] == "shortcut"
        )
        connection = sqlite3.connect(self.db)
        connection.execute(
            "UPDATE annotations SET direction = 'increase', confidence = 5 "
            "WHERE panel = 'shortcut' AND task_id = ?",
            (leaked_task,),
        )
        connection.commit()
        connection.close()
        result = self.validate()
        summary = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(summary["shortcut_gate"])


if __name__ == "__main__":
    unittest.main()
