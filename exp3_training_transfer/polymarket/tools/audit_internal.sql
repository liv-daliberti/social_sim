-- Internal integrity + coverage audit for polymarket_archive.
-- Sources: docs/polymarket_archive_completeness_recipe.md (1A, 2A, 3A-3C, 4) + coverage extras.
-- Run: apptainer exec --bind /n/fs/similarity postgres_16.sif psql "$URL" -v ON_ERROR_STOP=1 -P pager=off -f audit_internal.sql
\set ON_ERROR_STOP on
\timing off

\echo '=== [1A] event/market referential integrity ==='
WITH orphan_markets AS (
  SELECT COUNT(*) AS n
  FROM polymarket_markets m
  LEFT JOIN polymarket_events e ON e.event_id = m.event_id
  WHERE m.event_id IS NULL OR e.event_id IS NULL
),
events_without_markets AS (
  SELECT COUNT(*) AS n
  FROM polymarket_events e
  LEFT JOIN polymarket_markets m ON m.event_id = e.event_id
  WHERE m.market_id IS NULL
),
active_orphans AS (
  SELECT COUNT(*) AS n
  FROM polymarket_active_markets am
  LEFT JOIN polymarket_markets m ON m.condition_id = am.condition_id
  WHERE m.condition_id IS NULL
)
SELECT
  orphan_markets.n AS markets_missing_parent_event,
  events_without_markets.n AS events_with_zero_markets,
  active_orphans.n AS active_rows_missing_market
FROM orphan_markets, events_without_markets, active_orphans;

\echo '=== [2A] market quality ==='
SELECT
  COUNT(*) FILTER (WHERE condition_id IS NULL OR BTRIM(condition_id) = '') AS markets_missing_condition_id,
  COUNT(*) FILTER (
    WHERE clob_token_ids IS NULL
       OR jsonb_typeof(clob_token_ids) <> 'array'
       OR jsonb_array_length(clob_token_ids) = 0
  ) AS markets_missing_tokens,
  COUNT(*) FILTER (
    WHERE COALESCE(active, false) = true
      AND COALESCE(closed, false) = false
      AND NOT EXISTS (
        SELECT 1 FROM polymarket_active_markets am
        WHERE am.condition_id = m.condition_id
      )
  ) AS open_markets_not_in_active_table
FROM polymarket_markets m;

\echo '=== [4] baseline snapshot ==='
WITH market_tokens AS (
  SELECT m.market_id, token.token_id
  FROM polymarket_markets m
  LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(m.clob_token_ids, '[]'::jsonb)) token(token_id) ON TRUE
)
SELECT
  (SELECT COUNT(*) FROM polymarket_events) AS events,
  (SELECT COUNT(*) FROM polymarket_markets) AS markets,
  (SELECT COUNT(*) FROM polymarket_active_markets) AS active_markets,
  (SELECT COUNT(*) FROM market_tokens WHERE token_id IS NOT NULL) AS market_tokens,
  (SELECT COUNT(*) FROM polymarket_market_candles) AS candle_rows,
  (SELECT COUNT(*) FROM polymarket_market_candles WHERE period_interval_minutes = 1440) AS daily_candle_rows;

\echo '=== [C1] freshness: metadata + candles must reach the run date ==='
SELECT
  (SELECT MAX(created_time) FROM polymarket_markets) AS max_market_created,
  (SELECT MAX(updated_time) FROM polymarket_markets) AS max_market_updated,
  (SELECT MAX(created_time) FROM polymarket_events)  AS max_event_created,
  (SELECT MAX(end_period_ts) FROM polymarket_market_candles
    WHERE period_interval_minutes = 1440 AND end_period_ts <= NOW() + INTERVAL '1 day') AS max_past_daily_candle;

\echo '=== [C2] resolution coverage on closed markets ==='
SELECT
  COUNT(*) FILTER (WHERE COALESCE(closed, false)) AS closed_markets,
  COUNT(*) FILTER (
    WHERE COALESCE(closed, false)
      AND outcome_prices IS NOT NULL
      AND jsonb_typeof(outcome_prices) = 'array'
      AND jsonb_array_length(outcome_prices) > 0
  ) AS closed_with_outcome_prices,
  COUNT(*) FILTER (
    WHERE COALESCE(closed, false)
      AND EXISTS (
        SELECT 1 FROM jsonb_array_elements_text(outcome_prices) p
        WHERE p.value::numeric >= 0.999 OR p.value::numeric <= 0.001
      )
  ) AS closed_with_extreme_price
FROM polymarket_markets;

\echo '=== [C3] closed markets that ENDED in the gap window (must have metadata refresh) ==='
SELECT
  COUNT(*) AS gap_window_ended_markets,
  COUNT(*) FILTER (WHERE updated_time >= '2026-07-11') AS refreshed_today
FROM polymarket_markets
WHERE end_time >= '2026-05-01' AND end_time < '2026-07-12' AND COALESCE(closed, false);

\echo '=== [C4] daily candle coverage by market (candles reach market end or today) ==='
WITH m AS (
  SELECT
    m.market_id,
    m.condition_id,
    LEAST(COALESCE(m.end_time, NOW()), NOW()) AS effective_end
  FROM polymarket_markets m
  WHERE m.condition_id IS NOT NULL AND BTRIM(m.condition_id) <> ''
),
cov AS (
  SELECT c.condition_id, MAX(c.end_period_ts) AS max_daily
  FROM polymarket_market_candles c
  WHERE c.period_interval_minutes = 1440
  GROUP BY c.condition_id
)
SELECT
  COUNT(*) AS markets_with_condition,
  COUNT(cov.condition_id) AS markets_with_daily_candles,
  ROUND(100.0 * COUNT(cov.condition_id) / NULLIF(COUNT(*), 0), 2) AS pct_with_daily,
  COUNT(*) FILTER (WHERE cov.max_daily >= m.effective_end - INTERVAL '2 days') AS markets_daily_reaches_end,
  ROUND(100.0 * COUNT(*) FILTER (WHERE cov.max_daily >= m.effective_end - INTERVAL '2 days')
        / NULLIF(COUNT(*), 0), 2) AS pct_daily_reaches_end
FROM m LEFT JOIN cov ON cov.condition_id = m.condition_id;

\echo '=== [C5] duplicate candles across partitions (PK is per-partition) ==='
SELECT COUNT(*) AS duplicate_candle_keys FROM (
  SELECT token_id, period_interval_minutes, end_period_ts
  FROM polymarket_market_candles
  GROUP BY 1, 2, 3
  HAVING COUNT(*) > 1
) d;

\echo '=== [C6] memory-case coverage vs markets ==='
SELECT
  (SELECT COUNT(*) FROM market_memory_cases WHERE source = 'polymarket') AS memory_cases,
  (SELECT COUNT(*) FROM polymarket_markets WHERE condition_id IS NOT NULL) AS markets_with_condition,
  (SELECT COUNT(DISTINCT market_key) FROM market_memory_cases WHERE source = 'polymarket') AS distinct_case_keys;

\echo '=== [C7] gap-window created markets present ==='
SELECT
  date_trunc('month', created_time)::date AS month,
  COUNT(*) AS markets_created
FROM polymarket_markets
WHERE created_time >= '2026-04-01'
GROUP BY 1 ORDER BY 1;
