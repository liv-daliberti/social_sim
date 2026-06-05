"""Minimal Polymarket Gamma API client (no auth required)."""

from __future__ import annotations

from http.client import HTTPException, IncompleteRead, RemoteDisconnected
import json
import logging
import os
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_GAMMA_HOST = "https://gamma-api.polymarket.com"


class GammaApiError(RuntimeError):
    """Raised when a Gamma API call fails."""


def _gamma_host() -> str:
    return os.getenv("POLYMARKET_GAMMA_HOST", DEFAULT_GAMMA_HOST).rstrip("/")


def _build_url(path: str, params: dict[str, Any] | None = None) -> str:
    base = _gamma_host()
    if not path.startswith("/"):
        path = "/" + path
    url = base + path
    if params:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            url = f"{url}?{query}"
    return url


def _request_json(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> Any:
    url = _build_url(path, params)
    req = Request(
        url,
        headers={
            "User-Agent": "agentic-forecasting-ingestor/1.0",
            # Avoid keeping flaky chunked connections open for long-running backfills.
            "Connection": "close",
        },
    )
    max_attempts = max(1, int(os.getenv("POLYMARKET_GAMMA_REQUEST_RETRIES", "4")))
    base_backoff_s = max(0.0, float(os.getenv("POLYMARKET_GAMMA_RETRY_BACKOFF_SECONDS", "0.5")))
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
                raise GammaApiError(f"Gamma API request failed: {url}") from exc
            logger.warning(
                "Gamma API HTTP retryable error (status=%s attempt=%s/%s url=%s)",
                status,
                attempt,
                max_attempts,
                url,
            )
        except (IncompleteRead, RemoteDisconnected, HTTPException, URLError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "Gamma API transient read error (attempt=%s/%s url=%s error=%s)",
                attempt,
                max_attempts,
                url,
                exc.__class__.__name__,
            )

        backoff = base_backoff_s * (2 ** (attempt - 1))
        if backoff > 0:
            time.sleep(backoff)

    raise GammaApiError(
        f"Gamma API request failed after {max_attempts} attempts: {url}"
    ) from last_exc


def fetch_events_page(
    *,
    active: bool = True,
    closed: bool = False,
    limit: int = 100,
    offset: int = 0,
    series_id: int | None = None,
    tag_id: int | None = None,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch a single page of events."""
    payload = _request_json(
        "/events",
        {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "series_id": series_id,
            "tag_id": tag_id,
        },
        timeout=timeout,
    )
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("events") or payload.get("data") or payload.get("results")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def fetch_events_keyset_page(
    *,
    limit: int = 100,
    after_cursor: str | None = None,
    active: bool | None = None,
    closed: bool | None = None,
    series_id: int | None = None,
    tag_id: int | None = None,
    timeout: float = 10.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch a single cursor-paginated page of events."""
    params: dict[str, Any] = {
        "limit": limit,
        "after_cursor": after_cursor,
        "series_id": series_id,
        "tag_id": tag_id,
    }
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()

    payload = _request_json(
        "/events/keyset",
        params,
        timeout=timeout,
    )
    if isinstance(payload, dict):
        items = payload.get("events") or payload.get("data") or payload.get("results")
        next_cursor = payload.get("next_cursor")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)], (
                str(next_cursor) if next_cursor else None
            )
    return [], None


def fetch_markets_page(
    *,
    active: bool = True,
    closed: bool = False,
    limit: int = 100,
    offset: int = 0,
    event_id: int | None = None,
    series_id: int | None = None,
    tag_id: int | None = None,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch a single page of markets."""
    payload = _request_json(
        "/markets",
        {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "event_id": event_id,
            "series_id": series_id,
            "tag_id": tag_id,
        },
        timeout=timeout,
    )
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("markets") or payload.get("data") or payload.get("results")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def iter_pages(fetch_fn, *, limit: int, max_pages: int) -> Iterable[list[dict[str, Any]]]:
    """Yield pages from a Gamma endpoint that supports limit/offset."""
    offset = 0
    pages = 0
    while pages < max_pages:
        rows = fetch_fn(limit=limit, offset=offset)
        if not rows:
            return
        yield rows
        pages += 1
        if len(rows) < limit:
            return
        offset += limit


__all__ = [
    "GammaApiError",
    "fetch_events_page",
    "fetch_events_keyset_page",
    "fetch_markets_page",
    "iter_pages",
]
