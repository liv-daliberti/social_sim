"""Azure AI Foundry multi-turn structured forecast agent.

Uses azure-ai-projects >= 2.1.0 to call a named agent via the OpenAI
Responses API protocol (agent_reference pattern).

Multi-turn flow:
  Turn 1: event model + initial hypotheses
  Turn 2: web evidence gathering + updated hypothesis probs
  Turn 3 (optional): deepen on weak evidence areas
  Final:  produce structured JSON forecast

Returns a ForecastRecord with the full conversation trace and parsed forecast.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from prompts import (
    SYSTEM_PROMPT,
    TURN1_TEMPLATE,
    TURN2_TEMPLATE,
    TURN3_TEMPLATE,
    FINAL_TURN_TEMPLATE,
)


# ── config ─────────────────────────────────────────────────────────────────────

PROJECT_ENDPOINT = os.environ.get(
    "AZURE_PROJECT_ENDPOINT",
    "https://liv.services.ai.azure.com/api/projects/Liv-project",
)
AZURE_API_KEY  = os.environ.get("AZURE_AI_API_KEY", "")
AGENT_NAME     = os.environ.get("AZURE_AGENT_NAME", "forecasting-agent")
AGENT_VERSION  = os.environ.get("AZURE_AGENT_VERSION", "2")
DEFAULT_MODEL  = os.environ.get("AZURE_MODEL", "gpt-5.4")

# Azure AI Foundry Responses-API protocol uses "v1" as the api-version
_AZURE_API_VERSION = "v1"

_MAX_RETRIES = 3
_RETRY_BASE  = 4.0


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    turn: int
    user_content: str
    response_id: str
    output_text: str
    tool_calls: list[dict] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ForecastRecord:
    task_id: str
    market_id: str
    question: str
    description: str
    yes_price_market: float | None
    days_to_resolution: float | None
    category: str | None

    turns: list[Turn] = field(default_factory=list)
    structured_forecast: dict | None = None
    yes_prob: float | None = None

    parse_error: str | None = None
    error: str | None = None
    forecast_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "task_id":             self.task_id,
            "market_id":           self.market_id,
            "question":            self.question,
            "description":         self.description,
            "yes_price_market":    self.yes_price_market,
            "days_to_resolution":  self.days_to_resolution,
            "category":            self.category,
            "turns": [
                {
                    "turn":           t.turn,
                    "user_content":   t.user_content,
                    "response_id":    t.response_id,
                    "output_text":    t.output_text,
                    "tool_calls":     t.tool_calls,
                    "search_queries": t.search_queries,
                    "input_tokens":   t.input_tokens,
                    "output_tokens":  t.output_tokens,
                }
                for t in self.turns
            ],
            "structured_forecast":  self.structured_forecast,
            "yes_prob":             self.yes_prob,
            "parse_error":          self.parse_error,
            "error":                self.error,
            "forecast_at":          self.forecast_at,
            "total_input_tokens":   self.total_input_tokens,
            "total_output_tokens":  self.total_output_tokens,
            "n_turns":              len(self.turns),
        }


# ── client factory ─────────────────────────────────────────────────────────────

def make_openai_client(api_key: str | None = None, endpoint: str | None = None):
    """Return an OpenAI client pointed at the Azure AI Foundry agent endpoint.

    Azure AI Foundry Responses-API protocol:
      base_url = {project_endpoint}/agents/{agent_name}/endpoint/protocols/openai
      api-version = v1   (sent as query param on every request)
      auth header = api-key: <key>
    """
    key = api_key or AZURE_API_KEY
    if not key:
        raise ValueError(
            "API key not set.  Export AZURE_AI_API_KEY=<key> or pass --api-key."
        )
    ep = (endpoint or PROJECT_ENDPOINT).rstrip("/")
    base_url = f"{ep}/agents/{AGENT_NAME}/endpoint/protocols/openai"
    return OpenAI(
        base_url=base_url,
        api_key="placeholder",            # SDK requires non-empty; real auth via header
        default_headers={"api-key": key},
        default_query={"api-version": _AZURE_API_VERSION},
        max_retries=0,
    )


# ── response parsing helpers ───────────────────────────────────────────────────

def _extract_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    texts: list[str] = []
    for item in (getattr(response, "output", None) or []):
        if hasattr(item, "text"):
            texts.append(item.text)
        elif hasattr(item, "content"):
            for block in (item.content or []):
                if hasattr(block, "text"):
                    texts.append(block.text)
    return "\n".join(texts).strip()


def _extract_tool_calls(response) -> tuple[list[dict], list[str]]:
    calls: list[dict] = []
    queries: list[str] = []
    for item in (getattr(response, "output", None) or []):
        item_type = getattr(item, "type", None)
        if item_type in ("web_search_call", "tool_call"):
            call = {"type": item_type}
            if hasattr(item, "query"):
                call["query"] = item.query
                queries.append(item.query)
            if hasattr(item, "name"):
                call["name"] = item.name
            calls.append(call)
    return calls, queries


def _extract_usage(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)


_REFUSAL_PHRASES = (
    "i'm sorry, but i cannot",
    "i cannot assist with that",
    "i'm unable to assist",
    "i cannot help with",
    "i'm not able to assist",
    "i'm sorry, i cannot",
    "sorry, but i cannot",
)

_REFUSAL_PLACEHOLDER = (
    "[Research step: content policy restriction on this topic — "
    "model proceeding with prior knowledge and any available context.]"
)

def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    # check first 200 chars (refusal is always at the top) and full text
    return any(p in lowered[:200] for p in _REFUSAL_PHRASES)


def _parse_json_forecast(text: str) -> tuple[dict | None, str | None]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse failed: {e}"
    return None, "No JSON object found in final response"


# ── retry wrapper ──────────────────────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            wait = _RETRY_BASE * (2 ** attempt)
            print(f"\n    [rate-limit] sleeping {wait:.0f}s …", end="", flush=True)
            time.sleep(wait)
        except (APITimeoutError, APIError):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BASE * (2 ** attempt))
    raise RuntimeError("Max retries exceeded")


# ── agent reference extra_body ─────────────────────────────────────────────────

def _agent_ref(name: str = AGENT_NAME, version: str = AGENT_VERSION) -> dict:
    return {
        "agent_reference": {
            "name":    name,
            "version": version,
            "type":    "agent_reference",
        }
    }


# ── main forecast function ─────────────────────────────────────────────────────

def forecast_market(
    market: dict,
    *,
    client,
    model: str = DEFAULT_MODEL,
    agent_name: str = AGENT_NAME,
    agent_version: str = AGENT_VERSION,
    do_third_turn: bool = True,
    verbose: bool = False,
) -> ForecastRecord:
    """Run the multi-turn forecast for one market.  Returns a ForecastRecord."""

    rec = ForecastRecord(
        task_id            = market.get("task_id", f"pm_{market['market_id']}"),
        market_id          = market["market_id"],
        question           = market.get("question", ""),
        description        = market.get("description", ""),
        yes_price_market   = market.get("yes_price"),
        days_to_resolution = market.get("days_to_resolution"),
        category           = market.get("category"),
    )

    prev_id: str | None = None
    extra = _agent_ref(agent_name, agent_version)

    def _create(user_content: str) -> object:
        params: dict = dict(
            model=model,
            input=[{"role": "user", "content": user_content}],
            store=True,
            extra_body=extra,
        )
        if prev_id:
            params["previous_response_id"] = prev_id
        return _call_with_retry(client.responses.create, **params)

    # ── Turn 1: event model + hypotheses ───────────────────────────────────────
    t1_content = TURN1_TEMPLATE.format(
        question           = rec.question,
        description        = (rec.description or "(see question)")[:1500],
        yes_price          = rec.yes_price_market or 0.5,
        days_to_resolution = rec.days_to_resolution or 0,
        category           = rec.category or "unknown",
    )
    if verbose:
        print(f"\n  [T1] Sending event-model prompt …")

    r1 = _create(t1_content)
    t1_text = _extract_text(r1)
    t1_calls, t1_queries = _extract_tool_calls(r1)
    t1_in, t1_out = _extract_usage(r1)
    rec.turns.append(Turn(1, t1_content, r1.id, t1_text, t1_calls, t1_queries, t1_in, t1_out))
    prev_id = r1.id
    if verbose:
        print(f"     T1 done ({t1_out} out-tokens)")

    # ── Turn 2: evidence gathering ─────────────────────────────────────────────
    if verbose:
        print(f"  [T2] Researching …")
    r2 = _create(TURN2_TEMPLATE)
    t2_text = _extract_text(r2)
    t2_calls, t2_queries = _extract_tool_calls(r2)
    t2_in, t2_out = _extract_usage(r2)

    if _is_refusal(t2_text):
        if verbose:
            print(f"     T2 refusal detected — using placeholder, keeping search context")
        # Searches ran (via tool calls) even when text is blocked; the search
        # context is still in the Azure conversation via r2.id.  We keep prev_id
        # pointing at r2 so T3/Final can access those results, and store a
        # neutral placeholder instead of the refusal message.
        t2_text = _REFUSAL_PLACEHOLDER
    prev_id = r2.id

    rec.turns.append(Turn(2, TURN2_TEMPLATE, r2.id, t2_text, t2_calls, t2_queries, t2_in, t2_out))
    if verbose:
        print(f"     T2 done ({t2_out} out-tokens, {len(t2_queries)} searches)")

    # ── Turn 3 (optional): deepen ──────────────────────────────────────────────
    if do_third_turn:
        if verbose:
            print(f"  [T3] Refining estimates …")
        r3 = _create(TURN3_TEMPLATE)
        t3_text = _extract_text(r3)
        t3_calls, t3_queries = _extract_tool_calls(r3)
        t3_in, t3_out = _extract_usage(r3)

        if _is_refusal(t3_text):
            if verbose:
                print(f"     T3 refusal detected — using placeholder, keeping search context")
            t3_text = _REFUSAL_PLACEHOLDER
        prev_id = r3.id

        rec.turns.append(Turn(3, TURN3_TEMPLATE, r3.id, t3_text, t3_calls, t3_queries, t3_in, t3_out))
        if verbose:
            print(f"     T3 done ({t3_out} out-tokens, {len(t3_queries)} searches)")

    # ── Final turn: structured JSON forecast ───────────────────────────────────
    if verbose:
        print(f"  [T{len(rec.turns)+1}] Producing structured JSON …")
    r_final = _create(FINAL_TURN_TEMPLATE)
    tf_text = _extract_text(r_final)
    tf_calls, tf_queries = _extract_tool_calls(r_final)
    tf_in, tf_out = _extract_usage(r_final)
    rec.turns.append(Turn(
        len(rec.turns) + 1, FINAL_TURN_TEMPLATE, r_final.id,
        tf_text, tf_calls, tf_queries, tf_in, tf_out,
    ))

    parsed, err = _parse_json_forecast(tf_text)
    if parsed:
        rec.structured_forecast = parsed
        rec.yes_prob = float(parsed.get("yes_prob", 0))
    else:
        rec.parse_error = err

    if verbose:
        print(f"     Final done ({tf_out} out-tokens)  yes_prob={rec.yes_prob}")

    return rec
