from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from exp1_prospective.indirect_interventions.review_app.app import create_app
from exp1_prospective.indirect_interventions.scripts.audit_review_packet import audit


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data/development/pilot_v2_review_packet.json"
REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
CANDIDATES = ROOT / "data/development/pilot_candidates.jsonl"
MANIFEST = ROOT / "data/development/pilot_v2_review_manifest.json"
LEGACY_PACKET = ROOT / "data/development/pilot_review_packet.json"
LEGACY_REVIEWERS = ROOT / "data/development/pilot_reviewers.json"
LEGACY_MANIFEST = ROOT / "data/development/pilot_review_manifest.json"


def csrf(html: bytes) -> str:
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    if not match:
        raise AssertionError("CSRF token missing from response")
    return match.group(1).decode()


class PacketAuditTests(unittest.TestCase):
    def test_market_disjoint_packet_passes_audit(self) -> None:
        self.assertEqual(audit(CANDIDATES, PACKET, REVIEWERS, MANIFEST), [])

    def test_legacy_pilot_fails_repeated_market_audit(self) -> None:
        errors = audit(CANDIDATES, LEGACY_PACKET, LEGACY_REVIEWERS, LEGACY_MANIFEST)
        self.assertTrue(any("repeated market exposure" in error for error in errors))

    def test_shortcut_tasks_have_no_context_or_labels(self) -> None:
        packet = json.loads(PACKET.read_text())
        serialized = json.dumps(packet)
        for private in (
            "candidate_id", "market_id", "intended_label", "chain_nodes",
            "chain_edges", "initial_event_model", "increase_yes", "decrease_yes",
            "no_material_effect",
        ):
            self.assertNotIn(private, serialized)
        for task in packet["panels"]["shortcut"]:
            self.assertEqual(set(task), {"task_id", "panel", "market", "evidence"})


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tempdir.name) / "reviews.sqlite3"
        self.app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_TOKEN": "admin-test",
            "DATABASE": self.db,
            "PACKET_PATH": PACKET,
            "REVIEWERS_PATH": REVIEWERS,
        })
        self.client = self.app.test_client()
        self.reviewers = json.loads(REVIEWERS.read_text())["reviewers"]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def login(self, reviewer_index: int) -> bytes:
        page = self.client.get("/")
        response = self.client.post("/", data={
            "csrf_token": csrf(page.data),
            "access_code": self.reviewers[reviewer_index]["access_code"],
        })
        self.assertEqual(response.status_code, 302)
        return self.client.get("/review", follow_redirects=True).data

    def reviewer_index(self, panel: str) -> int:
        return next(
            index for index, reviewer in enumerate(self.reviewers)
            if reviewer["panel"] == panel
        )

    def test_full_context_submission_persists_and_exports(self) -> None:
        html = self.login(0)
        self.assertIn(b"Your task", html)
        self.assertIn(b"Treat the stated connection as true", html)
        self.assertIn(b"New information", html)
        self.assertIn(b"How it connects", html)
        self.assertIn(b"Makes YES more likely", html)
        self.assertIn(b"The connection is unclear", html)
        self.assertIn(b"How sure are you?", html)
        self.assertNotIn(b"Initial event model", html)
        self.assertNotIn(b"Uncertain variables", html)
        for private in (b"increase_yes", b"decrease_yes", b"no_material_effect"):
            self.assertNotIn(private, html)
        for removed_field in (
            b"Shortest causal chain",
            b"Mechanism nodes used",
            b"Plausibility of the evidence item",
            b"Expected update strength",
            b"How directly does the evidence text",
            b"Optional notes",
        ):
            self.assertNotIn(removed_field, html)

        task_id = self.reviewers[0]["task_ids"][0]
        bad = self.client.post(f"/review/{task_id}", data={"csrf_token": "wrong"})
        self.assertEqual(bad.status_code, 400)

        response = self.client.post(f"/review/{task_id}", data={
            "csrf_token": csrf(html),
            "direction": "increase",
            "confidence": "4",
        })
        self.assertEqual(response.status_code, 302)

        connection = sqlite3.connect(self.db)
        row = connection.execute(
            "SELECT panel, direction, confidence FROM annotations "
            "WHERE reviewer_id = ? AND task_id = ?",
            (self.reviewers[0]["reviewer_id"], task_id),
        ).fetchone()
        connection.close()
        self.assertEqual(row, ("full_context", "increase", 4))

        admin = self.client.get("/admin?token=admin-test")
        self.assertEqual(admin.status_code, 200)
        exported = self.client.get("/admin/export.json?token=admin-test")
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(len(exported.get_json()["annotations"]), 1)
        self.assertEqual(self.client.get("/admin").status_code, 404)

    def test_shortcut_panel_never_receives_bridge(self) -> None:
        shortcut_index = self.reviewer_index("shortcut")
        html = self.login(shortcut_index)
        self.assertIn(b"Your task", html)
        self.assertIn(b"Use only the new information shown", html)
        self.assertNotIn(b"Treat the stated connection as true", html)
        self.assertIn(b"How sure are you?", html)
        self.assertNotIn(b"Causal bridge", html)
        self.assertNotIn(b"How it connects", html)
        self.assertNotIn(b"Initial event model", html)
        task_id = self.reviewers[shortcut_index]["task_ids"][0]
        response = self.client.post(f"/review/{task_id}", data={
            "csrf_token": csrf(html),
            "direction": "ambiguous",
            "confidence": "4",
        })
        self.assertEqual(response.status_code, 302)
        connection = sqlite3.connect(self.db)
        row = connection.execute(
            "SELECT panel, direction, confidence FROM annotations"
        ).fetchone()
        connection.close()
        self.assertEqual(row, ("shortcut", "ambiguous", 4))

    def test_reviewer_cannot_open_unassigned_task(self) -> None:
        self.login(0)
        shortcut_task = self.reviewers[self.reviewer_index("shortcut")]["task_ids"][0]
        self.assertEqual(self.client.get(f"/review/{shortcut_task}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
