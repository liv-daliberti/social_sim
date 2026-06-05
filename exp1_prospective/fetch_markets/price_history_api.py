"""Polymarket CLOB price history client (no auth required)."""

from __future__ import annotations

from http.client import HTTPException, IncompleteRead, RemoteDisconnected
import json
import logging
import os
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


class PriceHistoryError(RuntimeError):
    """Raised when a price history request fails."""


def _clob_host() -> str:
    return os.getenv("POLYMARKET_CLOB_HOST", DEFAULT_CLOB_HOST).rstrip("/")


def _build_url(path: str, params: dict[str, Any] | None = None) -> str:
    base = _clob_host()
    if not path.startswith("/"):
        path = "/" + path
    url = base + path
    if params:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            url = f"{url}?{query}"
    return url


def _request_timeout_seconds(default: float = 30.0) -> float:
    raw = os.getenv("POLYMARKET_PRICE_HISTORY_REQUEST_TIMEOUT_SECONDS")
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def _request_json(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 15.0,
) -> Any:
    url = _build_url(path, params)
    req = Request(
        url,
        headers={
            "User-Agent": "agentic-forecasting-ingestor/1.0",
            "Connection": "close",
        },
    )
    max_attempts = max(1, int(os.getenv("POLYMARKET_PRICE_HISTORY_REQUEST_RETRIES", "4")))
    base_backoff_s = max(0.0, float(os.getenv("POLYMARKET_PRICE_HISTORY_RETRY_BACKOFF_SECONDS", "0.5")))
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            last_exc = exc
            status = getattr(exc, "code", None)
            retryable = status == 408 or status == 429 or (isinstance(status, int) and status >= 500)
            if not retryable or attempt >= max_attempts:
                raise PriceHistoryError(f"Polymarket price history request failed: {url}") from exc
            logger.warning(
                "Polymarket price history retryable HTTP error (status=%s attempt=%s/%s token=%s)",
                status,
                attempt,
                max_attempts,
                params.get("market") if isinstance(params, dict) else None,
            )
        except (
            IncompleteRead,
            RemoteDisconnected,
            HTTPException,
            TimeoutError,
            socket.timeout,
            URLError,
            json.JSONDecodeError,
        ) as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "Polymarket price history transient read error (attempt=%s/%s token=%s error=%s)",
                attempt,
                max_attempts,
                params.get("market") if isinstance(params, dict) else None,
                exc.__class__.__name__,
            )

        backoff = base_backoff_s * (2 ** (attempt - 1))
        if backoff > 0:
            time.sleep(backoff)

    raise PriceHistoryError(
        f"Polymarket price history request failed after {max_attempts} attempts: {url}"
    ) from last_exc


def _parse_price_history(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        items = (
            payload.get("history")
            or payload.get("prices")
            or payload.get("data")
            or payload.get("results")
        )
        if isinstance(items, list):
            return [item for item in items if item is not None]
    if isinstance(payload, list):
        return [item for item in payload if item is not None]
    return []


def _parse_history_row(row: Any) -> tuple[int | None, Any]:
    if row is None:
        return None, None
    if isinstance(row, dict):
        ts = (
            row.get("t")
            or row.get("ts")
            or row.get("timestamp")
            or row.get("time")
        )
        price = row.get("p")
        if price is None:
            price = row.get("price")
        return _coerce_int(ts), price
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return _coerce_int(row[0]), row[1]
    return None, None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def fetch_price_history(
    *,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_minutes: int,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Fetch price history for a token id.

    Returns list of dicts with keys: ts (epoch seconds), price (raw).
    """
    request_timeout = timeout if timeout is not None else _request_timeout_seconds()
    payload = _request_json(
        "/prices-history",
        {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity_minutes,
        },
        timeout=request_timeout,
    )
    rows = _parse_price_history(payload)
    points: list[dict[str, Any]] = []
    for row in rows:
        ts, price = _parse_history_row(row)
        if ts is None:
            continue
        points.append({"ts": ts, "price": price})
    points.sort(key=lambda item: item.get("ts") or 0)
    return points


def fetch_trades_page(
    *,
    token_id: str,
    after_ts: int,
    before_ts: int,
    limit: int = 500,
    cursor: str | None = None,
    timeout: float = 15.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one page of CLOB trades for a token.  Returns (trades, next_cursor)."""
    params: dict[str, Any] = {
        "market": token_id,
        "after": after_ts,
        "before": before_ts,
        "limit": limit,
    }
    if cursor:
        params["next_cursor"] = cursor
    try:
        payload = _request_json("/trades", params, timeout=timeout)
    except PriceHistoryError:
        return [], None
    if payload is None:
        return [], None
    if isinstance(payload, list):
        return [t for t in payload if isinstance(t, dict)], None
    if isinstance(payload, dict):
        data = payload.get("data") or []
        next_cursor = payload.get("next_cursor") or payload.get("nextCursor")
        return [t for t in data if isinstance(t, dict)], (
            str(next_cursor) if next_cursor else None
        )
    return [], None


def fetch_daily_volume(
    *,
    token_id: str,
    day_start_ts: int,
    day_end_ts: int,
    timeout: float = 15.0,
    delay: float = 0.5,
    max_pages: int = 50,
) -> float:
    """Return total USD volume for a token on a calendar day from CLOB trades.

    Sums price * size across all matched trades in [day_start_ts, day_end_ts].
    Returns 0.0 if the endpoint is unavailable or returns no data.
    """
    total = 0.0
    cursor: str | None = None

    for _page in range(max_pages):
        trades, next_cursor = fetch_trades_page(
            token_id=token_id,
            after_ts=day_start_ts,
            before_ts=day_end_ts,
            cursor=cursor,
            timeout=timeout,
        )
        for trade in trades:
            try:
                price = float(trade.get("price", 0) or 0)
                size  = float(trade.get("size",  0) or 0)
                total += price * size
            except (TypeError, ValueError):
                pass
        if not trades or not next_cursor:
            break
        cursor = next_cursor
        time.sleep(delay)

    return total


__all__ = [
    "DEFAULT_CLOB_HOST",
    "PriceHistoryError",
    "fetch_daily_volume",
    "fetch_price_history",
    "fetch_trades_page",
]
