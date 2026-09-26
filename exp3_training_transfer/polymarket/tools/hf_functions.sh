update_hf_readme() {
  local dataset_path="$1"
  local summary_path="${dataset_path%.json}.summary.json"
  "$PYTHON_BIN" - "$dataset_path" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
reports = dataset_path.parent
readme_path = reports / "polymarket_full_market_dataset_hf_README.md"
summary = json.loads(summary_path.read_text())

status_path = reports / "polymarket_daily_candles" / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}

event_gap_status_path = reports / "polymarket_event_coverage_audit" / "fill_status.json"
event_gap_status = json.loads(event_gap_status_path.read_text()) if event_gap_status_path.exists() else {}

market_status_path = reports / "polymarket_event_market_completeness" / "status.json"
market_status = json.loads(market_status_path.read_text()) if market_status_path.exists() else {}

all_summaries = sorted(
    reports.glob("polymarket_full_market_dataset_*.summary.json"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
previous = [path for path in all_summaries if path.resolve() != summary_path.resolve()]
previous_summary = json.loads(previous[0].read_text()) if previous else None
previous_dataset = previous[0].with_suffix("").with_suffix(".json").name if previous else None

def fmt_int(value):
    return f"{int(value):,}" if value is not None else "unknown"

def fmt_pct(value):
    return f"{float(value):.2f}%" if value is not None else "unknown"

latest_files = [
    dataset_path.name,
    summary_path.name,
    "polymarket_daily_candles_status.md",
    "polymarket_daily_candles_status.json",
    "polymarket_event_gap_fill_status.md",
    "polymarket_event_gap_fill_status.json",
    "polymarket_event_market_completeness_status.md",
    "polymarket_event_market_completeness_status.json",
    "polymarket_event_market_missing_markets.csv",
]

lines = [
    "# Polymarket Full Market Dataset",
    "",
    "This dataset is a market-centric JSON snapshot exported from the local Polymarket archive.",
    "",
    "## Latest Snapshot",
    "",
    "Files:",
    "",
]
lines.extend(f"- `{name}`" for name in latest_files)
lines.extend(
    [
        "",
        "Snapshot summary:",
        "",
        f"- Exported at: `{summary.get('generated_at')}`",
        f"- Markets: `{fmt_int(summary.get('row_count'))}`",
        f"- Markets with any candle rows in the exported snapshot: `{fmt_int(summary.get('markets_with_candles'))}`",
        f"- Total candle rows indexed in the exported snapshot: `{fmt_int(summary.get('total_candle_rows_indexed'))}`",
        "- Row ordering: `open_time ASC NULLS LAST`, then `market_created_time ASC`, then `market_id ASC`",
        "- Filters: none",
        "",
        "Dense daily candle backfill status at documentation update time:",
        "",
        f"- Updated: `{status.get('updated', 'unknown')}`",
        f"- Total events: `{fmt_int(status.get('total_events'))}`",
        f"- Total markets: `{fmt_int(status.get('total_markets'))}`",
        f"- Total CLOB tokens: `{fmt_int(status.get('total_tokens'))}`",
        f"- Events with any dense daily candles: `{fmt_int(status.get('events_with_any_daily'))}` (`{fmt_pct(status.get('event_any_pct'))}`)",
        f"- Events fully complete for dense daily candles: `{fmt_int(status.get('complete_events'))}` (`{fmt_pct(status.get('event_complete_pct'))}`)",
        f"- Tokens with dense daily candles: `{fmt_int(status.get('daily_tokens'))}` (`{fmt_pct(status.get('daily_token_pct'))}`)",
        f"- Markets with any dense daily candles: `{fmt_int(status.get('markets_with_any_daily'))}` (`{fmt_pct(status.get('market_any_pct'))}`)",
        f"- Markets fully complete for dense daily candles: `{fmt_int(status.get('complete_markets'))}` (`{fmt_pct(status.get('market_complete_pct'))}`)",
        f"- Dense daily candle rows currently stored: `{fmt_int(status.get('daily_rows'))}`",
        "",
        "Event gap fill status:",
        "",
        f"- Updated: `{event_gap_status.get('updated', 'unknown')}`",
        f"- Gamma rows scanned: `{fmt_int(event_gap_status.get('gamma_rows_scanned'))}`",
        f"- Missing events upserted on rerun: `{fmt_int(event_gap_status.get('missing_events_upserted'))}`",
        f"- Markets upserted on rerun: `{fmt_int(event_gap_status.get('markets_upserted'))}`",
        f"- Done: `{'YES' if event_gap_status.get('done') else event_gap_status.get('done', 'unknown')}`",
        "",
        "Event-to-market completeness repair status:",
        "",
        f"- Updated: `{market_status.get('updated', 'unknown')}`",
        f"- Gamma markets expected for local events: `{fmt_int(market_status.get('gamma_markets_expected'))}`",
        f"- Gamma markets already present: `{fmt_int(market_status.get('gamma_markets_already_present'))}`",
        f"- Missing Gamma markets upserted: `{fmt_int(market_status.get('missing_gamma_markets_upserted'))}`",
        f"- Events with missing markets repaired: `{fmt_int(market_status.get('events_with_missing_markets_repaired'))}`",
        f"- Done: `{'YES' if market_status.get('done') else market_status.get('done', 'unknown')}`",
        "",
        "The JSON snapshot is a point-in-time export. Dense daily candle backfill may continue to improve after this snapshot; use the status files to check coverage.",
    ]
)

if previous_summary and previous_dataset:
    lines.extend(
        [
            "",
            "## Previous Snapshot",
            "",
            "Files:",
            "",
            f"- `{previous_dataset}`",
            f"- `{previous[0].name}`",
            "",
            "Snapshot summary:",
            "",
            f"- Exported at: `{previous_summary.get('generated_at')}`",
            f"- Markets: `{fmt_int(previous_summary.get('row_count'))}`",
            f"- Markets with any candle rows in the exported snapshot: `{fmt_int(previous_summary.get('markets_with_candles'))}`",
            f"- Total candle rows indexed in the exported snapshot: `{fmt_int(previous_summary.get('total_candle_rows_indexed'))}`",
        ]
    )

lines.extend(
    [
        "",
        "## Row Contents",
        "",
        "Each market row contains:",
        "",
        "- `question` and `question_sources`",
        "- `rules`",
        "- `resolution`",
        "- `times`",
        "- `market`",
        "- `event`",
        "- `trade_metrics`",
        "- `order_book`",
        "- `market_closing_rules`",
        "- `market_memory`",
        "- `candles`",
        "- `candle_summary`",
        "- `availability`",
        "",
        f"Rows in `{dataset_path.name}` are ordered by model-facing `open_time` ascending, then `market_created_time` ascending, then `market_id` ascending. This is suitable for a coarse chronological pass over markets, but it is still one row per market and includes future supervision fields such as final resolution.",
        "",
        "For leakage-aware training/evaluation, prefer a temporal panel generated by `scripts/export_polymarket_temporal_panel.py`.",
        "",
        "## Candle Coverage Caveat",
        "",
        "Market metadata coverage is much stronger than candle coverage. The archive contains broad Polymarket market, event, rule, outcome, and timestamp metadata, while dense daily OHLC candle history is being filled incrementally.",
        "",
        "For dense daily candle rows:",
        "",
        "- `period_interval_minutes = 1440`",
        "- one row is targeted per CLOB token per UTC day in each market window",
        "- rows may be synthetic carry-forward rows or empty placeholder rows when no trade point exists for that day",
        "- `candles[].raw_meta` records the source/empty markers when available",
        "",
        "Use the snapshot summary and `polymarket_daily_candles_status.*` files to check candle completeness before relying on time-series coverage.",
        "",
    ]
)

readme_path.write_text("\n".join(lines), encoding="utf-8")
print(readme_path)
PY
}
upload_to_hf() {
  local dataset_path="$1"
  local summary_path="${dataset_path%.json}.summary.json"
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.hf"
  set +a
  export HF_HOME="$ROOT/.runtime/hf_upload_cache"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
  "$HF_PYTHON_BIN" - "$HF_REPO_ID" "$dataset_path" "$summary_path" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = sys.argv[1]
dataset_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
root = dataset_path.parents[1]
reports = root / "reports"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
api = HfApi(token=token)

uploads = [
    (dataset_path, dataset_path.name),
    (summary_path, summary_path.name),
    (reports / "polymarket_full_market_dataset_hf_README.md", "README.md"),
    (reports / "polymarket_daily_candles" / "status.md", "polymarket_daily_candles_status.md"),
    (reports / "polymarket_daily_candles" / "status.json", "polymarket_daily_candles_status.json"),
    (reports / "polymarket_event_coverage_audit" / "fill_status.md", "polymarket_event_gap_fill_status.md"),
    (reports / "polymarket_event_coverage_audit" / "fill_status.json", "polymarket_event_gap_fill_status.json"),
    (reports / "polymarket_event_market_completeness" / "status.md", "polymarket_event_market_completeness_status.md"),
    (reports / "polymarket_event_market_completeness" / "status.json", "polymarket_event_market_completeness_status.json"),
    (reports / "polymarket_event_market_completeness" / "missing_markets.csv", "polymarket_event_market_missing_markets.csv"),
]

for local_path, repo_path in uploads:
    if not local_path.exists():
        print(f"skip missing {local_path}", flush=True)
        continue
    print(f"upload {local_path} -> {repo_id}/{repo_path}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update Polymarket snapshot {dataset_path.stem}",
    )
PY
}
