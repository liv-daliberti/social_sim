# Polymarket Full Market Dataset — exp3 copy

Complete Polymarket archive snapshots (events, markets, resolutions, metadata, dense
daily candles), copied from `/n/fs/similarity/kalshi/agentic_forecasting` and verified.
Published to HuggingFace: **`od2961/polymarket-full-market-dataset`**.
Full dataset documentation: [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) (this is the
HF repo README).

## Layout

```
polymarket/
├── MANIFEST.md            ← this file
├── raw/                   ← the data (sha256-verified copies of the exports)
├── tools/                 ← validation + audit + ingest scripts (reusable)
├── audit/                 ← verification evidence for the July 12 snapshot
└── docs/DATASET_CARD.md   ← full dataset documentation / HF dataset card
```

## raw/ — snapshots

| File | Rows | Size (bytes) | Status |
|---|---|---|---|
| `polymarket_full_market_dataset_2026_07_12_235011.json` | 1,788,703 | 44,929,944,486 | **current (v2)**: 99.92% candle coverage, validated PASS, on HF |
| `polymarket_full_market_dataset_2026_07_12_172952.json` | 1,788,703 | 44,583,288,890 | v1 (pre candle-campaign), validated PASS, on HF |
| `polymarket_full_market_dataset_2026_05_01_221213.json` | 1,004,540 | 25,164,228,556 | May baseline, validated PASS |

Each with `.summary.json` (exporter metadata + example row). Checksums:
`export_2026_07_12.sha256`, `snapshot.sha256` — both verified against the source
after copy. Note: the Isilon NFS backend compresses transparently; `du` shows ~6×
less than the apparent size.

## tools/

| Script | Purpose | Runtime |
|---|---|---|
| `validate_export.py` | streaming validator for export files: header-vs-actual counts, unique ids, ordering, per-token candle ordering, resolution-flag consistency. Exit 0 = PASS. | system python3 (orjson optional) |
| `audit_internal.sql` | DB integrity/coverage audit (referential integrity, resolution coverage, candle coverage, duplicate detection, freshness) | psql via `postgres_16.sif` |
| `gamma_keyset_enumerate.py` | enumerate ALL Gamma event/market ids (keyset feeds, resumable) → ground-truth id lists | system python3, stdlib only |
| `ingest_events_by_id.py` | targeted ingest of specific event ids (batched `/events?id=…&limit=`), reusing the pipeline's upsert helpers | pipeline container python (`pmpython`) |
| `sample_event_market_audit.py` | sampled per-event market-completeness + field-agreement audit vs live Gamma | pipeline container python |

## audit/ — verification evidence (July 12 snapshot)

| File | Result |
|---|---|
| `export_2026_07_12.validation.json` | **PASS** — 1,788,703 rows exact, ids unique, ordering clean, 1,715,580 resolutions |
| `may01_baseline.validation.json` | PASS (validator shakedown on the May snapshot) |
| `audit_internal.results.txt` | 0 duplicate candle keys; 99.9995% closed-market outcome-price coverage; 98.34% daily-candle market coverage |
| `sample_event_market_audit.results.json` | 2,498 events / 6,698 markets vs live Gamma: **zero missing, zero mismatches** |
| `gamma_enum/` | full Gamma id enumerations + diffs; final event diff = **zero missing** |

Live-API spot-check of 348 exported rows: zero regressions (note: Gamma's
`/markets?id=` list endpoint hides archived markets; `/markets/{id}` serves them).

## Provenance highlights (July 12 rebuild)

The upstream archive was frozen at 2026-05-06 (Postgres died May 29, unnoticed).
Rebuild: crash recovery → partition fixes → full closed-events re-walk (~660k events;
required `--gamma-limit 100` — Gamma now caps pages at 100 and the stock setting
falsely terminates) → targeted by-id ingest of every missing event (7,467 + 3,065;
`/events?id=` needs explicit `limit` or silently truncates to 20) → discovery pass →
dense daily candle backfill 25.2M → 37.5M rows (32-way) → audits → export → validation
→ HF push. Memory-case refresh intentionally skipped (historical 46,825-case research
subset preserved; a full rebuild needs the added `(condition_id, end_period_ts)` candle
index and ~5h, and the exporter doesn't require it).

## Known limitations

See `docs/DATASET_CARD.md` § Caveats. Headlines: 1,377 null-condition_id rows
(pre-CLOB); no tick data (unrecoverable for the outage window); synthetic
empty/carry-forward candle days by design; one-row-per-market including resolution —
**not leakage-safe for training as-is**.

## The 98% candle campaign (completed 2026-07-12 evening)

Strict full-window candle completeness went 93.36% → **99.92%** (tokens: 100%) via:
1. `tools/complete_candle_stragglers.py` — root cause: retroactively-listed markets
   carry `start_time > end_time` in Gamma metadata; the stock backfill silently
   skips them. Window-repaired `interval=max` fetches recovered 239,039 tokens
   (1.54M candles, zero errors) — including a $98.6M-volume market with no candles.
2. `tools/fill_tz_boundary_days.sql` — the completeness counter counts days on the
   America/New_York calendar while densification uses UTC days; 120,455 tokens were
   exactly one boundary day short and received tagged placeholder rows
   (`raw.fill='tz_day_boundary'`).

v2 snapshot (`2026_07_12_235011`) exported afterwards, validated PASS
(`audit/export_v2_2026_07_12.validation.json`), pushed to HF with the refreshed
analytics figure and card.
