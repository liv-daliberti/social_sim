-- Fill timezone-boundary day gaps in dense daily candles.
--
-- The completeness counter (render_polymarket_daily_candle_status.py) counts
-- expected days on the SESSION-timezone (America/New_York) calendar, while the
-- backfill densifies UTC days — leaving some tokens one row short per boundary.
-- This inserts explicit empty placeholder rows (tagged raw.fill='tz_day_boundary')
-- for Eastern-calendar days inside each incomplete token's nominal window that
-- have no existing row. Pure DB operation; idempotent (ON CONFLICT DO NOTHING).
\set ON_ERROR_STOP on
BEGIN;
SET LOCAL statement_timeout = '3600s';
SET LOCAL work_mem = '2GB';

CREATE TEMP TABLE _mt ON COMMIT DROP AS
SELECT
  token.token_id,
  DATE_TRUNC('day', COALESCE(m.start_time, e.start_time, m.created_time, e.created_time, m.end_time, e.end_time)) AS day_start,
  DATE_TRUNC('day', LEAST(COALESCE(m.end_time, e.end_time, NOW()), NOW())) AS day_end,
  GREATEST(1, DATE_PART('day',
      DATE_TRUNC('day', LEAST(COALESCE(m.end_time, e.end_time, NOW()), NOW()))
    - DATE_TRUNC('day', COALESCE(m.start_time, e.start_time, m.created_time, e.created_time, m.end_time, e.end_time))
  )::int + 1) AS expected_days
FROM polymarket_markets m
LEFT JOIN polymarket_events e ON e.event_id = m.event_id
JOIN LATERAL jsonb_array_elements_text(COALESCE(m.clob_token_ids, '[]'::jsonb)) token(token_id) ON TRUE
WHERE m.clob_token_ids IS NOT NULL AND jsonb_typeof(m.clob_token_ids) = 'array'
  AND COALESCE(m.start_time, e.start_time, m.created_time, e.created_time, m.end_time, e.end_time) IS NOT NULL;

CREATE TEMP TABLE _have ON COMMIT DROP AS
SELECT token_id, COUNT(*) AS have_days
FROM polymarket_market_candles WHERE period_interval_minutes = 1440 GROUP BY token_id;
CREATE INDEX ON _have (token_id);

CREATE TEMP TABLE _incomplete ON COMMIT DROP AS
SELECT mt.token_id, MIN(mt.day_start) AS day_start, MAX(mt.day_end) AS day_end
FROM _mt mt LEFT JOIN _have h ON h.token_id = mt.token_id
WHERE COALESCE(h.have_days, 0) < mt.expected_days
GROUP BY mt.token_id;

SELECT 'incomplete_tokens', COUNT(*) FROM _incomplete;

CREATE TEMP TABLE _tokattr ON COMMIT DROP AS
SELECT DISTINCT ON (token_id) token_id, condition_id, outcome
FROM polymarket_market_candles
WHERE period_interval_minutes = 1440
  AND token_id IN (SELECT token_id FROM _incomplete);

INSERT INTO polymarket_market_candles
  (token_id, condition_id, outcome, period_interval_minutes, start_period_ts, end_period_ts,
   open, high, low, close, volume, liquidity, raw)
SELECT
  i.token_id, a.condition_id, a.outcome, 1440,
  d.day,
  d.day + interval '1 day' - interval '1 microsecond',
  NULL, NULL, NULL, NULL, NULL, NULL,
  jsonb_build_object(
    'source', 'polymarket_prices_history_dense_daily',
    'empty', true,
    'empty_reason', 'no_prior_trade',
    'fill', 'tz_day_boundary'
  )
FROM _incomplete i
JOIN _tokattr a ON a.token_id = i.token_id
CROSS JOIN LATERAL generate_series(i.day_start, i.day_end, interval '1 day') AS d(day)
LEFT JOIN polymarket_market_candles c
  ON c.token_id = i.token_id
 AND c.period_interval_minutes = 1440
 AND DATE_TRUNC('day', c.end_period_ts) = d.day
WHERE c.token_id IS NULL
ON CONFLICT (token_id, period_interval_minutes, end_period_ts) DO NOTHING;

SELECT 'rows_inserted', COUNT(*) FROM polymarket_market_candles
WHERE period_interval_minutes = 1440 AND raw->>'fill' = 'tz_day_boundary';

COMMIT;
