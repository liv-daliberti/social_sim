"""Web search tool for local model agents.

Primary backend: Tavily (AI-optimized, set TAVILY_API_KEY).
Fallback backend: DuckDuckGo (free, no key required, rate-limited).

Usage:
    tool = WebSearchTool()                      # auto-selects backend
    tool = WebSearchTool(backend="tavily")      # force Tavily (key required)
    tool = WebSearchTool(backend="duckduckgo")  # force DDG
    results = tool.search("US election 2026")
"""

from __future__ import annotations

import json
import os
import time
from typing import Literal


# ── OpenAI-compatible tool schema ──────────────────────────────────────────────

WEB_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information about a topic. "
            "Use this to find recent news, official statements, polling data, "
            "or factual context relevant to the prediction market you are forecasting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string. Be specific and targeted.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

TOOLS: list[dict] = [WEB_SEARCH_TOOL]


# ── WebSearchTool ──────────────────────────────────────────────────────────────

class WebSearchTool:
    """Web search with Tavily (primary) and DuckDuckGo (fallback)."""

    def __init__(
        self,
        backend: Literal["auto", "tavily", "duckduckgo"] = "auto",
        max_chars_per_result: int = 800,
        default_max_results: int = 5,
        retry_delay: float = 2.0,
    ):
        self.max_chars_per_result = max_chars_per_result
        self.default_max_results = default_max_results
        self.retry_delay = retry_delay
        self._requested_backend = backend
        self._active_backend: str = ""
        self._tavily_client = None
        self._setup()

    def _setup(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY", "")

        if self._requested_backend in ("auto", "tavily") and api_key:
            try:
                from tavily import TavilyClient  # type: ignore
                self._tavily_client = TavilyClient(api_key=api_key)
                self._active_backend = "tavily"
                return
            except ImportError:
                if self._requested_backend == "tavily":
                    raise RuntimeError(
                        "tavily-python is not installed. Run: pip install tavily-python"
                    )

        if self._requested_backend == "tavily":
            raise RuntimeError(
                "Tavily backend requested but TAVILY_API_KEY is not set."
            )

        # DuckDuckGo fallback
        try:
            __import__("duckduckgo_search")
            self._active_backend = "duckduckgo"
        except ImportError:
            raise RuntimeError(
                "No search backend is available. Install one of:\n"
                "  pip install tavily-python      # set TAVILY_API_KEY\n"
                "  pip install duckduckgo-search   # free, no key needed"
            )

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Search and return a list of {title, url, content} dicts."""
        n = max_results if max_results is not None else self.default_max_results
        if self._active_backend == "tavily":
            return self._search_tavily(query, n)
        return self._search_duckduckgo(query, n)

    def _search_tavily(self, query: str, n: int) -> list[dict]:
        resp = self._tavily_client.search(
            query=query,
            max_results=n,
            search_depth="basic",
            include_answer=False,
        )
        results = []
        for r in resp.get("results", []):
            content = (r.get("content") or r.get("snippet") or "")[: self.max_chars_per_result]
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": content,
            })
        return results

    def _search_duckduckgo(self, query: str, n: int) -> list[dict]:
        from duckduckgo_search import DDGS  # type: ignore
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n):
                    content = (r.get("body") or "")[: self.max_chars_per_result]
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("href", ""),
                        "content": content,
                    })
        except Exception:
            # DDG is rate-limited; back off and return empty rather than crash
            time.sleep(self.retry_delay)
        return results

    def format_for_model(self, results: list[dict]) -> str:
        """Format search results as a compact string for injection as a tool message."""
        if not results:
            return "No results found for this query."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"{r['content']}"
            )
        return "\n\n".join(parts)

    def execute_tool_call(self, tool_call) -> str:
        """Execute an OpenAI tool_call object and return formatted result string."""
        try:
            args = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            return "Error: could not parse tool call arguments."
        query = args.get("query", "").strip()
        if not query:
            return "Error: empty query."
        max_results = args.get("max_results", self.default_max_results)
        results = self.search(query, max_results)
        return self.format_for_model(results)
