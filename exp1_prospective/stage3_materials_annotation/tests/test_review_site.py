import csv
import io
import json
import sqlite3
from pathlib import Path

from exp1_prospective.stage3_materials_annotation.review_site.app import create_app


ROOT = Path(__file__).resolve().parents[1]


def make_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_TOKEN": "test-admin",
            "DATABASE": tmp_path / "reviews.sqlite3",
            "PUBLIC_ITEMS": ROOT / "generated_v6/public_items.jsonl",
            "ASSIGNMENTS": ROOT / "generated_v6/assignments.json",
        }
    )


def post_with_csrf(client, path, data):
    with client.session_transaction() as session:
        session["csrf_token"] = "test-csrf"
    return client.post(path, data={"csrf_token": "test-csrf", **data})


def login_and_consent(client, code):
    response = post_with_csrf(client, "/", {"access_code": code})
    assert response.status_code == 302
    response = post_with_csrf(
        client, "/consent", {"adult_consent": "yes"}
    )
    assert response.status_code == 302


def valid_rating():
    return {
        "conditional_direction": "more_likely",
        "direction_confidence": "4",
        "clarity": "5",
        "plausibility": "4",
        "usable_premise": "yes",
    }


def exported_rows(client, cohort):
    response = client.get(f"/export/{cohort}.csv?token=test-admin")
    assert response.status_code == 200
    return list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))


def test_practice_flow_is_separate_from_registered_export(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    login_and_consent(client, "self")

    response = client.get("/review", follow_redirects=True)
    assert response.status_code == 200
    assert b"Self-review practice" in response.data
    assert b"June 10, 2026 is today" in response.data
    assert b"Information available on June 10" in response.data
    assert b"Item 1 of 18" in response.data

    item_id = app.extensions["annotation_reviewers"]["self_review"]["item_ids"][0]
    response = post_with_csrf(client, f"/review/{item_id}", valid_rating())
    assert response.status_code == 302

    practice = exported_rows(client, "practice")
    registered = exported_rows(client, "registered")
    assert len(practice) == 1
    assert practice[0]["reviewer_id"] == "self_review"
    assert practice[0]["cohort"] == "practice"
    assert registered == []


def test_registered_flow_never_enters_practice_export(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    login_and_consent(client, "annotator_01")
    item_id = app.extensions["annotation_reviewers"]["annotator_01"]["item_ids"][0]
    response = post_with_csrf(client, f"/review/{item_id}", valid_rating())
    assert response.status_code == 302

    registered = exported_rows(client, "registered")
    practice = exported_rows(client, "practice")
    assert len(registered) == 1
    assert registered[0]["reviewer_id"] == "annotator_01"
    assert registered[0]["cohort"] == "registered"
    assert practice == []

    connection = sqlite3.connect(tmp_path / "reviews.sqlite3")
    cohorts = connection.execute(
        "SELECT reviewer_id, cohort FROM annotations"
    ).fetchall()
    connection.close()
    assert cohorts == [("annotator_01", "registered")]


def test_form_requires_all_registered_fields(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()
    login_and_consent(client, "self")
    item_id = app.extensions["annotation_reviewers"]["self_review"]["item_ids"][0]

    response = post_with_csrf(
        client,
        f"/review/{item_id}",
        {**valid_rating(), "conditional_direction": ""},
    )
    assert response.status_code == 200
    assert b"Select a conditional direction" in response.data
    assert exported_rows(client, "practice") == []


def test_practice_only_deployment_rejects_registered_codes(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_TOKEN": "test-admin",
            "PRACTICE_ONLY": True,
            "DATABASE": tmp_path / "reviews.sqlite3",
            "PUBLIC_ITEMS": ROOT / "generated_v6/public_items.jsonl",
            "ASSIGNMENTS": ROOT / "generated_v6/assignments.json",
        }
    )
    client = app.test_client()

    response = post_with_csrf(client, "/", {"access_code": "annotator_01"})
    assert response.status_code == 200
    assert b"not recognized" in response.data

    response = post_with_csrf(client, "/", {"access_code": "self"})
    assert response.status_code == 302


def test_production_uses_private_codes_and_disables_practice(tmp_path):
    codes = {
        f"annotator_{index:02d}": f"review-code-{index:02d}-private"
        for index in range(1, 8)
    }
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_TOKEN": "test-admin",
            "PRODUCTION": True,
            "ENABLE_PRACTICE": False,
            "PRACTICE_ONLY": False,
            "REVIEWER_CODES_JSON": json.dumps(codes),
            "DATABASE": tmp_path / "reviews.sqlite3",
            "PUBLIC_ITEMS": ROOT / "generated_v6/public_items.jsonl",
            "ASSIGNMENTS": ROOT / "generated_v6/assignments.json",
        }
    )
    client = app.test_client()

    response = post_with_csrf(client, "/", {"access_code": "annotator_01"})
    assert response.status_code == 200
    assert b"not recognized" in response.data

    response = post_with_csrf(client, "/", {"access_code": "self"})
    assert response.status_code == 200
    assert b"not recognized" in response.data

    response = post_with_csrf(client, "/", {"access_code": codes["annotator_01"]})
    assert response.status_code == 302


def test_health_is_private_and_admin_status_is_token_gated(tmp_path):
    app = make_app(tmp_path)
    client = app.test_client()

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Frame-Options"] == "DENY"

    assert client.get("/admin/status").status_code == 403
    response = client.get("/admin/status?token=test-admin")
    assert response.status_code == 200
    assert response.get_json()["expected_ratings"] == 126
