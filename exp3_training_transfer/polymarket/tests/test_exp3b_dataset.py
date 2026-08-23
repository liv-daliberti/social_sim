#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_exp3b_dataset as builder  # noqa: E402


def make_row() -> dict:
    close = datetime(2026, 3, 15, tzinfo=timezone.utc)
    decision = close - timedelta(days=builder.HORIZON_DAYS)
    candles = []
    for offset, probability in enumerate((0.31, 0.34, 0.36, 0.40, 0.43)):
        timestamp = decision - timedelta(days=5 - offset)
        candles.append(
            {
                "outcome": "Yes",
                "period_interval_minutes": 1440,
                "end_period_ts": builder.iso(timestamp),
                "close": probability,
                "raw_meta": {"source": "real"},
            }
        )
    candles.extend(
        [
            {
                "outcome": "Yes",
                "period_interval_minutes": 1440,
                "end_period_ts": builder.iso(decision),
                "close": 0.99,
                "raw_meta": {"carry_forward": True},
            },
            {
                "outcome": "Yes",
                "period_interval_minutes": 1440,
                "end_period_ts": builder.iso(decision + timedelta(days=1)),
                "close": 0.91,
                "raw_meta": {"source": "future"},
            },
            {
                "outcome": "No",
                "period_interval_minutes": 1440,
                "end_period_ts": builder.iso(decision),
                "close": 0.57,
                "raw_meta": {"source": "real"},
            },
        ]
    )
    return {
        "ids": {"market_id": 17, "condition_id": "0xabc", "event_id": 9},
        "question": "Will the Senate pass the bill by March 2026?",
        "rules": {"market_description": "Resolves YES if the bill passes."},
        "outcomes": {"is_binary": True, "labels": ["Yes", "No"]},
        "resolution": {
            "winner_yes": True,
            "resolution_method": "price_extreme",
            "resolution_time": builder.iso(close),
        },
        "times": {
            "open_time": builder.iso(decision - timedelta(days=60)),
            "close_time": builder.iso(close),
            "market_end_time": builder.iso(close),
        },
        "event": {
            "title": "Senate bill vote",
            "category": "Politics",
            "series_id": "44",
            "tags": [{"label": "Politics", "slug": "politics"}],
        },
        "candles": candles,
    }


class DatasetBuilderTest(unittest.TestCase):
    def test_censors_future_and_synthetic_prices(self) -> None:
        skipped = builder.Counter()
        task = builder.task_from_row(make_row(), skipped)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(len(task["price_history"]), 5)
        self.assertAlmostEqual(task["market_yes_prob"], 0.43)
        self.assertNotIn("0.9100", task["input"])
        self.assertNotIn("0.9900", task["input"])
        self.assertNotIn("winner_yes", task["input"])
        self.assertNotIn("settlement_yes", task["input"])

    def test_keeps_only_latest_genuine_close_per_utc_date(self) -> None:
        row = make_row()
        duplicate = dict(row["candles"][0])
        duplicate["end_period_ts"] = duplicate["end_period_ts"].replace(
            "00:00:00Z", "12:00:00Z"
        )
        duplicate["close"] = 0.33
        row["candles"].append(duplicate)
        task = builder.task_from_row(row, builder.Counter())
        assert task is not None
        self.assertEqual(len(task["price_history"]), 5)
        self.assertAlmostEqual(task["price_history"][0]["yes_close"], 0.33)

    def test_domain_filter_rejects_lexical_false_positives(self) -> None:
        row = make_row()
        row["question"] = "Will X be nominated for Best Picture?"
        row["event"] = {
            "title": "Academy Award nominations",
            "category": "Awards",
            "tags": [{"label": "Awards", "slug": "awards"}],
        }
        self.assertIsNone(builder.task_from_row(row, builder.Counter()))

        row["question"] = 'Will Nebius say "ClickHouse" during its earnings call?'
        row["event"] = {
            "title": "Nebius earnings call mentions",
            "category": "Finance",
            "tags": [{"label": "Finance", "slug": "finance"}],
        }
        self.assertIsNone(builder.task_from_row(row, builder.Counter()))

    def test_reference_is_separate_from_prompt(self) -> None:
        task = builder.task_from_row(make_row(), builder.Counter())
        assert task is not None
        self.assertIn('"settlement_yes": true', task["reference"])
        self.assertNotIn(task["reference"], task["input"])

    def test_family_keys_cover_event_series_and_template(self) -> None:
        task = builder.task_from_row(make_row(), builder.Counter())
        assert task is not None
        keys = builder.leakage_keys(task)
        self.assertIn("event:9", keys)
        self.assertIn("series:44", keys)
        self.assertTrue(any(key.startswith("template:") for key in keys))

    def test_template_normalizes_dates_and_numbers(self) -> None:
        first = builder.template_key("Will X happen by March 15, 2026?")
        second = builder.template_key("Will X happen by April 20, 2027?")
        self.assertEqual(first, second)

    def test_training_schedule_is_complete_and_balanced(self) -> None:
        rows = [{"task_id": f"task-{index}"} for index in range(7)]
        schedule = builder.make_training_schedule(rows)
        counts = builder.Counter(row["task_id"] for row in schedule)
        self.assertEqual(len(schedule), builder.TRAIN_PRESENTATIONS)
        self.assertEqual(set(counts), {row["task_id"] for row in rows})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_rejects_non_yes_no_binary(self) -> None:
        row = make_row()
        row["outcomes"]["labels"] = ["Candidate A", "Candidate B"]
        skipped = builder.Counter()
        self.assertIsNone(builder.task_from_row(row, skipped))
        self.assertEqual(skipped["not_yes_no_binary"], 1)


if __name__ == "__main__":
    unittest.main()
