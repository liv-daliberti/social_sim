from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from build_tasks import anchors_for, scenario_ends_for  # noqa: E402
from common import STUDY, load_frozen_tasks  # noqa: E402
from select_probe import ridge_fit, ridge_predict  # noqa: E402


def test_frozen_tasks_are_balanced_and_donor_is_sealed():
    rows, manifest = load_frozen_tasks()
    assert manifest["study"] == STUDY
    assert len(rows) == 512
    counts = Counter((row["split"], row["k"], row["variant"]) for row in rows)
    assert counts[("dev", 0, "original")] == 96
    assert counts[("test", 8, "transplant")] == 32
    by_episode = {row["episode_id"]: row for row in rows}
    for row in rows:
        donor = by_episode[row["donor_episode_id"]]
        assert donor["split"] == row["split"]
        assert donor["donor_episode_id"] == row["episode_id"]


def test_transplant_changes_only_selected_reference_trajectory():
    rows, _ = load_frozen_tasks()
    pairs = {}
    for row in rows:
        pairs.setdefault((row["episode_id"], row["k"]), {})[row["variant"]] = row
    for variants in pairs.values():
        original, transplant = variants["original"], variants["transplant"]
        assert original["truth_targets"] == transplant["truth_targets"]
        assert original["prompt"] != transplant["prompt"]
        assert original["anchor_char_ends"].keys() == transplant["anchor_char_ends"].keys()
        assert len(original["patch_scenario_char_ends"]) == 10


def test_raw_prompt_sites_are_well_ordered():
    rows, _ = load_frozen_tasks()
    for row in rows[::31]:
        anchors = anchors_for(row["prompt"])
        assert list(anchors.values()) == sorted(anchors.values())
        sites = scenario_ends_for(row["prompt"])
        assert sites == sorted(sites)
        assert sites[-1] == anchors["query_end"]


def test_common_ridge_recovers_synthetic_linear_target():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 12))
    beta = rng.normal(size=(12, 2))
    y = x @ beta + np.asarray((0.2, -0.7))
    fitted = ridge_fit(x, y, 1e-8)
    prediction = ridge_predict(fitted, x)
    assert np.mean((prediction - y) ** 2) < 1e-10


def test_protocol_is_explicitly_paper_external():
    protocol = json.loads((HERE / "protocol" / "frozen_protocol.json").read_text())
    assert protocol["paper_status"] == "paper_external_do_not_render_or_import_into_manuscript"
    assert len(protocol["confirmatory_endpoints"]) == 6
    assert protocol["replication_rule"].startswith("submit the complete Qwen3-4B")
