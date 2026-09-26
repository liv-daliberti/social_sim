#!/usr/bin/env python3
"""Build the registered, time-censored Experiment 3B Polymarket benchmark.

One example is created per resolved YES/NO market at a fixed 30-day horizon.
Only genuine daily YES-token observations at or before the decision timestamp are
shown. Carry-forward candles, final prices, final volumes, status flags, and all
resolution fields are excluded from model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson
except ImportError:  # OAT environment; the Tinker environment has the fast path.
    orjson = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "raw" / "polymarket_full_market_dataset_2026_07_12_235011.json"
DEFAULT_OUTPUT = ROOT / "data" / "exp3b_registered"
EXPECTED_SNAPSHOT_SHA256 = "5697f7672bcb8dcc0af73272ff9ff1ec662241e92b0ac84f6e2bd4ba8f638e7d"
PROTOCOL_VERSION = "exp3b_registered_v1"
HORIZON_DAYS = 30
MAX_STALENESS_DAYS = 7
HISTORY_POINTS = 14
TRAIN_PRESENTATIONS = 4_800
MIN_REAL_PRICES = 4
TRAIN_END = datetime(2025, 7, 1, tzinfo=timezone.utc)
DEV_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

POLITICAL_HINTS = (
    "politics", "political", "election", "elections", "presidential", "president",
    "senate", "senator", "house of representatives", "congress",
    "governor", "republican", "democrat", "primary election",
    "party nominee", "white house", "supreme court", "cabinet",
    "parliament", "prime minister", "referendum", "ballot", "trump",
    "biden", "harris", "government",
)
GEOPOLITICAL_HINTS = (
    "geopolitics", "geopolitical", "ceasefire", "sanction", "nato",
    "ukraine", "russia", "israel", "gaza", "iran", "taiwan", "sanctions",
    "north korea", "military strike", "invasion", "peace deal",
)
MACRO_HINTS = (
    "federal reserve", "fed rate", "fed rates", "interest rate", "interest rates",
    "recession", "unemployment", "nonfarm payroll", "gdp", "tariff", "tariffs",
    "macro indicator", "economic growth", "government shutdown",
)
HARD_EXCLUDED_CATEGORIES = {
    "awards", "culture", "movies", "music", "sports", "esports", "crypto",
}
DENY_TAGS = {
    "sports", "esports", "tennis", "crypto", "bitcoin", "ethereum",
    "solana", "xrp", "up or down", "nba", "nfl", "nhl", "mlb", "soccer",
    "golf", "ufc", "formula 1", "pga", "wnba", "chess", "games",
}
MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SPACE_RE = re.compile(r"\s+")


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def clean_text(value: Any, limit: int) -> str:
    text = SPACE_RE.sub(" ", str(value or "")).strip()
    return text[:limit]


def primary_rules(row: dict[str, Any]) -> str:
    rules = row.get("rules") or {}
    if isinstance(rules, str):
        return clean_text(rules, 3_000)
    for key in ("market_description", "primary_rules_text", "event_description"):
        if rules.get(key):
            return clean_text(rules[key], 3_000)
    return ""


def tags(row: dict[str, Any]) -> list[str]:
    event = row.get("event") or {}
    values = []
    for tag in event.get("tags") or []:
        if isinstance(tag, dict):
            values.extend((str(tag.get("label") or ""), str(tag.get("slug") or "")))
        else:
            values.append(str(tag))
    values.extend((str(event.get("category") or ""), str(event.get("slug") or "")))
    return [value.strip().lower() for value in values if value.strip()]


def contains_phrase(haystack: str, phrase: str) -> bool:
    tokens = [re.escape(token) for token in phrase.lower().split()]
    pattern = r"(?<![a-z])" + r"\s+".join(tokens) + r"(?![a-z])"
    return re.search(pattern, haystack) is not None


def judgment_domain(row: dict[str, Any]) -> str | None:
    row_tags = tags(row)
    if any(value in DENY_TAGS for value in row_tags):
        return None
    event = row.get("event") or {}
    category = str(event.get("category") or "").strip().lower()
    if category in HARD_EXCLUDED_CATEGORIES:
        return None
    haystack = " ".join(
        (
            str(row.get("question") or ""),
            str(event.get("title") or ""),
            " ".join(row_tags),
        )
    ).lower()
    if "up or down" in haystack:
        return None
    for domain, hints in (
        ("politics_government", POLITICAL_HINTS),
        ("geopolitics", GEOPOLITICAL_HINTS),
        ("macroeconomics", MACRO_HINTS),
    ):
        if any(contains_phrase(haystack, hint) for hint in hints):
            return domain
    return None

def template_key(question: str) -> str:
    text = MONTH_RE.sub("<month>", question.lower())
    text = NUMBER_RE.sub("<n>", text)
    text = re.sub(r"[^a-z<>]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def terminal_time(row: dict[str, Any]) -> datetime | None:
    resolution = row.get("resolution") or {}
    times = row.get("times") or {}
    candidates = [
        parse_ts(resolution.get("resolution_time")),
        parse_ts(times.get("close_time")),
        parse_ts(times.get("market_end_time")),
        parse_ts(times.get("event_end_time")),
    ]
    candidates = [value for value in candidates if value is not None]
    return min(candidates) if candidates else None


def genuine_yes_history(
    row: dict[str, Any], decision_ts: datetime
) -> list[dict[str, Any]]:
    by_date: dict[str, tuple[datetime, float]] = {}
    for candle in row.get("candles") or []:
        if str(candle.get("outcome") or "").strip().lower() != "yes":
            continue
        if int(candle.get("period_interval_minutes") or 0) != 1440:
            continue
        raw_meta = candle.get("raw_meta") or {}
        if raw_meta.get("empty") or raw_meta.get("carry_forward"):
            continue
        close = candle.get("close")
        candle_ts = parse_ts(candle.get("end_period_ts"))
        if close is None or candle_ts is None or candle_ts > decision_ts:
            continue
        try:
            probability = float(close)
        except (TypeError, ValueError):
            continue
        if 0.0 <= probability <= 1.0:
            date = candle_ts.date().isoformat()
            previous = by_date.get(date)
            if previous is None or candle_ts > previous[0]:
                by_date[date] = (candle_ts, probability)
    return [
        {"ts": iso(timestamp), "yes_close": round(probability, 6)}
        for timestamp, probability in sorted(by_date.values())
    ]


def build_prompt(task: dict[str, Any]) -> str:
    history = "\n".join(
        f"- {point['ts'][:10]}: {float(point['yes_close']):.4f}"
        for point in task["price_history"]
    )
    rules = task["rules"] or "No additional resolution rules were archived."
    return (
        "You are forecasting a historical binary prediction-market question. "
        "Use only the information below as it stood at the forecast cutoff.\n\n"
        f"Forecast cutoff (UTC): {task['decision_ts']}\n"
        f"Question: {task['question']}\n"
        f"Resolution rules: {rules}\n\n"
        "Genuine pre-cutoff YES closing prices (oldest to newest):\n"
        f"{history}\n\n"
        "Return exactly one JSON object with your probability that the market "
        "resolves YES: {\"yes_prob\": <number from 0 to 1>}\n/no_think"
    )


def task_from_row(row: dict[str, Any], skipped: Counter[str]) -> dict[str, Any] | None:
    outcomes = row.get("outcomes") or {}
    labels = [str(label).strip().lower() for label in outcomes.get("labels") or []]
    if not outcomes.get("is_binary") or labels != ["yes", "no"]:
        skipped["not_yes_no_binary"] += 1
        return None
    resolution = row.get("resolution") or {}
    if resolution.get("resolution_method") != "price_extreme":
        skipped["not_price_extreme_resolution"] += 1
        return None
    settlement = resolution.get("winner_yes")
    if not isinstance(settlement, bool):
        skipped["missing_boolean_settlement"] += 1
        return None
    domain = judgment_domain(row)
    if domain is None:
        skipped["outside_judgment_domain"] += 1
        return None

    close_ts = terminal_time(row)
    open_ts = parse_ts((row.get("times") or {}).get("open_time"))
    if close_ts is None or open_ts is None:
        skipped["missing_open_or_close_time"] += 1
        return None
    decision_ts = close_ts - timedelta(days=HORIZON_DAYS)
    if open_ts > decision_ts - timedelta(days=1):
        skipped["not_open_before_cutoff"] += 1
        return None
    if decision_ts >= TEST_END or close_ts > datetime(2026, 7, 13, tzinfo=timezone.utc):
        skipped["outside_registered_time_window"] += 1
        return None

    history = genuine_yes_history(row, decision_ts)
    if len(history) < MIN_REAL_PRICES:
        skipped["fewer_than_four_real_prices"] += 1
        return None
    latest_ts = parse_ts(history[-1]["ts"])
    assert latest_ts is not None
    if decision_ts - latest_ts > timedelta(days=MAX_STALENESS_DAYS):
        skipped["stale_market_price"] += 1
        return None
    market_probability = float(history[-1]["yes_close"])
    if not 0.02 <= market_probability <= 0.98:
        skipped["already_effectively_resolved"] += 1
        return None

    ids = row.get("ids") or {}
    event = row.get("event") or {}
    market_id = str(ids.get("condition_id") or ids.get("market_id") or "")
    event_id = str(ids.get("event_id") or "")
    series_id = str(event.get("series_id") or "")
    question = clean_text(row.get("question") or event.get("title"), 700)
    if not market_id or not event_id or not question:
        skipped["missing_identifier_or_question"] += 1
        return None
    question_template = template_key(question)
    task = {
        "task_id": f"polymarket::{market_id}::{iso(decision_ts)}",
        "market_id": market_id,
        "event_id": event_id,
        "series_id": series_id,
        "template_key": question_template,
        "question": question,
        "rules": primary_rules(row),
        "category": str(event.get("category") or "").strip().lower(),
        "domain": domain,
        "decision_ts": iso(decision_ts),
        "market_close_ts": iso(close_ts),
        "horizon_days": HORIZON_DAYS,
        "price_history": history[-HISTORY_POINTS:],
        "market_yes_prob": market_probability,
        "settlement_yes": settlement,
    }
    task["input"] = build_prompt(task)
    task["reference"] = json.dumps(
        {
            "task_id": task["task_id"],
            "market_id": market_id,
            "event_id": event_id,
            "settlement_yes": settlement,
            "market_yes_prob": market_probability,
            "decision_ts": task["decision_ts"],
        },
        sort_keys=True,
    )
    return task


def iter_archive(
    path: Path, digest: Any, max_rows: int = 0
) -> Iterable[tuple[dict[str, Any] | None, str | None]]:
    count = 0
    in_rows = False
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            stripped = line.strip()
            if not in_rows:
                if stripped == b'"rows": [':
                    in_rows = True
                continue
            if stripped in {b"]", b"],", b"}"}:
                continue
            payload = stripped.rstrip(b",")
            # Reject exact schema failures before decoding large candle arrays.
            if b'"labels": ["Yes", "No"]' not in payload:
                yield None, "not_yes_no_binary"
            elif b'"resolution_method": "price_extreme"' not in payload:
                yield None, "not_price_extreme_resolution"
            else:
                row = orjson.loads(payload) if orjson is not None else json.loads(payload)
                yield row, None
            count += 1
            if max_rows and count >= max_rows:
                return


def split_name(task: dict[str, Any]) -> str | None:
    decision = parse_ts(task["decision_ts"])
    assert decision is not None
    if decision < TRAIN_END:
        return "train"
    if decision < DEV_END:
        return "dev"
    if decision < TEST_END:
        return "test"
    return None


def leakage_keys(task: dict[str, Any]) -> set[str]:
    keys = {f"event:{task['event_id']}", f"template:{task['template_key']}"}
    if task["series_id"]:
        keys.add(f"series:{task['series_id']}")
    return keys


def stable_sample(rows: list[dict[str, Any]], cap: int, salt: str) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{PROTOCOL_VERSION}|{salt}|{row['task_id']}".encode()
        ).hexdigest(),
    )
    return ordered[:cap] if cap > 0 else ordered


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "yes_rate": None, "first_decision_ts": None, "last_decision_ts": None}
    dates = sorted(row["decision_ts"] for row in rows)
    return {
        "count": len(rows),
        "yes_rate": sum(bool(row["settlement_yes"]) for row in rows) / len(rows),
        "mean_market_yes_prob": sum(float(row["market_yes_prob"]) for row in rows) / len(rows),
        "first_decision_ts": dates[0],
        "last_decision_ts": dates[-1],
        "distinct_events": len({row["event_id"] for row in rows}),
        "distinct_series": len({row["series_id"] for row in rows if row["series_id"]}),
        "distinct_templates": len({row["template_key"] for row in rows}),
    }

def make_training_schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot create a training schedule without rows")
    schedule: list[dict[str, Any]] = []
    epoch = 0
    while len(schedule) < TRAIN_PRESENTATIONS:
        remaining = TRAIN_PRESENTATIONS - len(schedule)
        schedule.extend(stable_sample(rows, 0, f"train-schedule-{epoch}")[:remaining])
        epoch += 1
    return schedule


def save_hf(
    output: Path, splits: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    from datasets import Dataset, DatasetDict, Features, Value

    features = Features({"input": Value("string"), "reference": Value("string")})

    def save(name: str, rows: list[dict[str, Any]]) -> None:
        model_rows = [{"input": row["input"], "reference": row["reference"]} for row in rows]
        dataset = (
            Dataset.from_list(model_rows, features=features)
            if model_rows
            else Dataset.from_dict({"input": [], "reference": []}, features=features)
        )
        DatasetDict({"train": dataset}).save_to_disk(str(output / name))

    for split, rows in splits.items():
        save(f"hf_{split}", rows)
    schedule = make_training_schedule(splits["train"])
    save("hf_train_schedule", schedule)
    return schedule


def build(args: argparse.Namespace) -> dict[str, Any]:
    skipped: Counter[str] = Counter()
    candidates: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    digest = hashlib.sha256()
    parsed_rows = 0
    for row, fast_skip in iter_archive(args.snapshot, digest, args.max_rows):
        parsed_rows += 1
        if parsed_rows % 100_000 == 0:
            print(f"parsed {parsed_rows:,} rows; retained {sum(map(len, candidates.values())):,}", flush=True)
        if fast_skip:
            skipped[fast_skip] += 1
            continue
        assert row is not None
        task = task_from_row(row, skipped)
        if task is None:
            continue
        split = split_name(task)
        if split is None:
            skipped["outside_split_dates"] += 1
            continue
        candidates[split].append(task)

    snapshot_hash = digest.hexdigest()
    if not args.max_rows and snapshot_hash != EXPECTED_SNAPSHOT_SHA256:
        raise AssertionError(
            f"snapshot SHA-256 mismatch: expected {EXPECTED_SNAPSHOT_SHA256}, got {snapshot_hash}"
        )

    train_keys = set().union(*(leakage_keys(row) for row in candidates["train"]))
    dev = [row for row in candidates["dev"] if not (leakage_keys(row) & train_keys)]
    skipped["dev_seen_family"] += len(candidates["dev"]) - len(dev)
    dev_keys = set().union(*(leakage_keys(row) for row in dev)) if dev else set()
    earlier_keys = train_keys | dev_keys
    test = [row for row in candidates["test"] if not (leakage_keys(row) & earlier_keys)]
    skipped["test_seen_family"] += len(candidates["test"]) - len(test)

    selected = {
        "train": stable_sample(candidates["train"], args.train_cap, "train"),
        "dev": stable_sample(dev, args.dev_cap, "dev"),
        "test": stable_sample(test, args.test_cap, "test"),
    }
    if not args.max_rows:
        minimums = {"train": 1000, "dev": 128, "test": 128}
        for split, minimum in minimums.items():
            if len(selected[split]) < minimum:
                raise AssertionError(f"{split} has only {len(selected[split])} tasks; expected at least {minimum}")

    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in selected.items():
        write_jsonl(args.output / f"{split}.tasks.jsonl", rows)
    training_schedule = save_hf(args.output, selected)

    selected_key_sets = {
        split: set().union(*(leakage_keys(row) for row in rows)) if rows else set()
        for split, rows in selected.items()
    }
    overlap = {
        "train_dev": len(selected_key_sets["train"] & selected_key_sets["dev"]),
        "train_test": len(selected_key_sets["train"] & selected_key_sets["test"]),
        "dev_test": len(selected_key_sets["dev"] & selected_key_sets["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"family leakage after selection: {overlap}")

    outputs = {
        f"{split}.tasks.jsonl": file_sha256(args.output / f"{split}.tasks.jsonl")
        for split in selected
    }
    manifest = {
        "status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "source": {
            "path": str(args.snapshot),
            "sha256": snapshot_hash,
            "expected_sha256": EXPECTED_SNAPSHOT_SHA256,
            "fully_scanned": not bool(args.max_rows),
            "parsed_rows": parsed_rows,
        },
        "design": {
            "one_cutoff_per_market": True,
            "horizon_days": HORIZON_DAYS,
            "minimum_genuine_prices": MIN_REAL_PRICES,
            "maximum_price_staleness_days": MAX_STALENESS_DAYS,
            "history_points_in_prompt": HISTORY_POINTS,
            "training_prompt_presentations": TRAIN_PRESENTATIONS,
            "labels": "price_extreme YES/NO resolutions only",
            "domains": "politics, government, macroeconomics, and geopolitics",
            "train": "decision_ts < 2025-07-01",
            "dev": "2025-07-01 <= decision_ts < 2026-01-01",
            "test": "2026-01-01 <= decision_ts < 2026-07-01",
            "cross_split_exclusions": ["event_id", "series_id", "normalized_question_template"],
            "forbidden_model_fields": [
                "resolution", "latest_prices", "final volume", "closed/active flags",
                "post-cutoff candles", "carry-forward candles",
            ],
        },
        "candidate_counts_before_family_filter": {
            split: len(rows) for split, rows in candidates.items()
        },
        "selected": {split: describe(rows) for split, rows in selected.items()},
        "training_schedule": {
            "prompt_presentations": len(training_schedule),
            "unique_markets": len(selected["train"]),
            "minimum_repetitions": min(Counter(row["task_id"] for row in training_schedule).values()),
            "maximum_repetitions": max(Counter(row["task_id"] for row in training_schedule).values()),
            "task_order_sha256": hashlib.sha256(
                "\n".join(row["task_id"] for row in training_schedule).encode()
            ).hexdigest(),
        },
        "cross_split_key_overlap": overlap,
        "skipped": dict(sorted(skipped.items())),
        "output_sha256": outputs,
        "known_limitation": (
            "The archive has no revision history for static question/rules text; this protocol "
            "treats those listing fields as time-invariant and censors every timestamped field."
        ),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--train-cap", type=int, default=4_800)
    result.add_argument("--dev-cap", type=int, default=512)
    result.add_argument("--test-cap", type=int, default=1_024)
    result.add_argument("--max-rows", type=int, default=0, help="Smoke-test prefix; disables full hash/minimum checks")
    return result


if __name__ == "__main__":
    build(parser().parse_args())
