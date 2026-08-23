"""Local model structured forecast agent (Qwen2.5-7B / Llama3.1-8B via Ollama).

Uses Ollama's OpenAI-compatible API with an explicit ReAct tool-calling loop
for web search, vs. the Azure AI Foundry agent_reference used by frontier agents.

Multi-turn flow:
  Turn 1 (no tools):  event model + initial hypotheses
  Turn 2 (tool loop): research phase — model calls web_search until satisfied
  Turn 3 (optional, tool loop): red/blue team steelman
  Final (no tools, json_object mode): structured JSON forecast + repair if needed

The ForecastRecord and Turn dataclasses are reused from forecast_agent.py so
the output schema is identical to the frontier agents.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# load .env from the same directory if present (picks up TAVILY_API_KEY etc.)
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from prompts import (
    SYSTEM_PROMPT,
    TURN1_TEMPLATE,
    TURN2_TEMPLATE,
    TURN3_TEMPLATE,
    FINAL_TURN_TEMPLATE,
)
from local_tools import WebSearchTool, TOOLS
from forecast_agent import Turn, ForecastRecord, _parse_json_forecast


# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MODEL           = "qwen2.5:7b"

_MAX_RETRIES = 3
_RETRY_BASE  = 4.0

_JSON_REPAIR_PROMPT = (
    "Your previous response was not valid JSON or could not be parsed. "
    "Output ONLY the JSON object — no markdown fences, no text before or after. "
    "Start your response with { and end with }."
)


# ── client factory ─────────────────────────────────────────────────────────────

def make_local_client(endpoint: str = DEFAULT_OLLAMA_ENDPOINT, timeout: float = 600.0) -> OpenAI:
    """Return an OpenAI client pointed at a local Ollama server.

    For large models (70b+) on CPU, set timeout to 3600+ seconds.
    """
    return OpenAI(
        base_url=endpoint,
        api_key="ollama",   # Ollama ignores this; SDK requires non-empty string
        max_retries=0,
        timeout=timeout,
    )


# ── retry wrapper ──────────────────────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            wait = _RETRY_BASE * (2 ** attempt)
            time.sleep(wait)
        except (APITimeoutError, APIError):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BASE * (2 ** attempt))
    raise RuntimeError("Max retries exceeded")


# ── tool call helpers ──────────────────────────────────────────────────────────

def _tool_calls_to_dicts(tool_calls) -> tuple[list[dict], list[str]]:
    """Convert OpenAI tool_call objects to serialisable dicts for Turn storage."""
    call_dicts: list[dict] = []
    queries: list[str] = []
    for tc in (tool_calls or []):
        d: dict = {"type": "tool_call", "name": tc.function.name, "id": tc.id}
        try:
            args = json.loads(tc.function.arguments)
            d["arguments"] = args
            if "query" in args:
                d["query"] = args["query"]
                queries.append(args["query"])
        except json.JSONDecodeError:
            d["raw_arguments"] = tc.function.arguments
        call_dicts.append(d)
    return call_dicts, queries


def _assistant_msg_with_tools(msg) -> dict:
    """Serialise an OpenAI assistant message (with tool_calls) back to a dict."""
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id":       tc.id,
                "type":     "function",
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


# ── pre-search helpers ─────────────────────────────────────────────────────────

def _pre_search_queries(question: str, n: int = 3) -> list[str]:
    """Generate n search queries from the market question."""
    q = question.rstrip("?.,;").strip()
    return [
        q,
        q + " latest news 2026",
        q + " expert forecast probability",
    ][:n]


def _run_pre_searches(
    messages: list[dict],
    question: str,
    search_tool: WebSearchTool,
    n: int,
    verbose: bool,
) -> list[str]:
    """Execute n searches and inject results into messages as a single context block.

    Small models (7-8B) reliably ignore tool_choice="required" and answer
    without searching. This guarantees search results are in context before
    the synthesis call, matching the information access frontier models get.
    Returns the list of queries executed.
    """
    queries = _pre_search_queries(question, n)
    parts: list[str] = []
    for i, q in enumerate(queries):
        result = search_tool.search(q)
        parts.append(f"Search {i + 1}: {q!r}\n{result}")
        if verbose:
            print(f"       search ({i + 1}/{n}): {q[:80]}")
    # Bundle into one user message so the model sees them before it synthesises
    messages.append({
        "role":    "user",
        "content": "Here are web search results gathered for this question:\n\n"
                   + "\n\n".join(parts)
                   + "\n\nPlease use these results in your analysis.",
    })
    return queries


# ── ReAct tool loop ────────────────────────────────────────────────────────────

def _run_tool_loop(
    messages: list[dict],
    *,
    client: OpenAI,
    model: str,
    search_tool: WebSearchTool,
    max_search_calls: int,
    max_tokens: int,
    temperature: float,
    verbose: bool,
) -> tuple[str, list[dict], list[str], int, int]:
    """ReAct loop: call model, execute tool calls, repeat until text response.

    Modifies messages in place (appends assistant + tool messages).
    Returns (final_text, all_tool_call_dicts, all_queries, total_in_tokens, total_out_tokens).
    """
    all_call_dicts: list[dict] = []
    all_queries:    list[str]  = []
    total_in  = 0
    total_out = 0
    search_count = 0

    while True:
        response = _call_with_retry(
            client.chat.completions.create,
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        msg     = response.choices[0].message
        total_in  += getattr(response.usage, "prompt_tokens", 0)
        total_out += getattr(response.usage, "completion_tokens", 0)

        if msg.tool_calls and search_count < max_search_calls:
            messages.append(_assistant_msg_with_tools(msg))
            for tc in msg.tool_calls:
                result_text = search_tool.execute_tool_call(tc)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })
                search_count += 1
                if verbose:
                    try:
                        q = json.loads(tc.function.arguments).get("query", "?")
                    except Exception:
                        q = tc.function.arguments[:80]
                    print(f"       extra search ({search_count}): {q[:80]}")
            call_dicts, queries = _tool_calls_to_dicts(msg.tool_calls)
            all_call_dicts.extend(call_dicts)
            all_queries.extend(queries)
        else:
            # model produced a text response (or search cap reached)
            final_text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": final_text})
            break

    return final_text, all_call_dicts, all_queries, total_in, total_out


# ── main forecast function ─────────────────────────────────────────────────────

def forecast_market_local(
    market: dict,
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    search_tool: WebSearchTool,
    do_third_turn: bool = True,
    max_search_calls: int = 6,
    max_tokens: int = 1500,
    temperature: float = 0.1,
    json_repair_attempts: int = 1,
    verbose: bool = False,
) -> ForecastRecord:
    """Run the multi-turn local forecast for one market. Returns a ForecastRecord."""

    rec = ForecastRecord(
        task_id            = market.get("task_id", f"pm_{market['market_id']}"),
        market_id          = market["market_id"],
        question           = market.get("question", ""),
        description        = market.get("description", ""),
        yes_price_market   = market.get("yes_price"),
        days_to_resolution = market.get("days_to_resolution"),
        category           = market.get("category"),
    )

    days = rec.days_to_resolution or 0
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Turn 1: event model (no tools) ────────────────────────────────────────
    t1_content = TURN1_TEMPLATE.format(
        question           = rec.question,
        description        = (rec.description or "(see question)")[:1500],
        yes_price          = rec.yes_price_market or 0.5,
        days_to_resolution = days,
        category           = rec.category or "unknown",
    )
    messages.append({"role": "user", "content": t1_content})

    if verbose:
        print("  [T1] Building event model …")

    r1       = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    t1_text  = (r1.choices[0].message.content or "").strip()
    t1_in    = getattr(r1.usage, "prompt_tokens", 0)
    t1_out   = getattr(r1.usage, "completion_tokens", 0)
    messages.append({"role": "assistant", "content": t1_text})
    rec.turns.append(Turn(1, t1_content, "", t1_text, [], [], t1_in, t1_out))

    if verbose:
        print(f"     T1 done ({t1_out} tokens)")

    # ── Turn 2: evidence gathering (pre-search + optional tool loop) ─────────────
    t2_content = TURN2_TEMPLATE.format(days_to_resolution=days)
    messages.append({"role": "user", "content": t2_content})

    if verbose:
        print("  [T2] Researching …")

    # Guarantee at least 3 searches regardless of whether the model cooperates
    # with tool calling (small models often ignore tool_choice entirely).
    pre_queries = _run_pre_searches(messages, rec.question, search_tool, n=3, verbose=verbose)

    # Allow the model to add further searches (up to remaining budget) via the loop
    t2_text, t2_calls, t2_extra, t2_in, t2_out = _run_tool_loop(
        messages,
        client=client,
        model=model,
        search_tool=search_tool,
        max_search_calls=max(0, max_search_calls - len(pre_queries)),
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )
    t2_queries = pre_queries + t2_extra
    rec.turns.append(Turn(2, t2_content, "", t2_text, t2_calls, t2_queries, t2_in, t2_out))

    if verbose:
        print(f"     T2 done ({len(t2_queries)} searches)")

    # ── Turn 3 (optional): red/blue team (tool loop, remaining budget) ────────
    if do_third_turn:
        remaining_searches = max(0, max_search_calls - len(t2_queries))
        messages.append({"role": "user", "content": TURN3_TEMPLATE})

        if verbose:
            print("  [T3] Red/blue team …")

        t3_text, t3_calls, t3_queries, t3_in, t3_out = _run_tool_loop(
            messages,
            client=client,
            model=model,
            search_tool=search_tool,
            max_search_calls=remaining_searches,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        rec.turns.append(Turn(3, TURN3_TEMPLATE, "", t3_text, t3_calls, t3_queries, t3_in, t3_out))

        if verbose:
            print(f"     T3 done ({len(t3_queries)} searches)")

    # ── Final turn: structured JSON (json_object mode, no tools) ─────────────
    fn = len(rec.turns) + 1
    messages.append({"role": "user", "content": FINAL_TURN_TEMPLATE})

    if verbose:
        print(f"  [T{fn}] Producing structured JSON …")

    rf = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    tf_text = (rf.choices[0].message.content or "").strip()
    tf_in   = getattr(rf.usage, "prompt_tokens", 0)
    tf_out  = getattr(rf.usage, "completion_tokens", 0)

    parsed, err = _parse_json_forecast(tf_text)

    # one repair attempt if parse failed
    if parsed is None and json_repair_attempts > 0:
        if verbose:
            print(f"     JSON parse failed ({err}) — attempting repair …")
        messages.append({"role": "assistant", "content": tf_text})
        messages.append({"role": "user", "content": _JSON_REPAIR_PROMPT})
        rf2 = _call_with_retry(
            client.chat.completions.create,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        tf_text = (rf2.choices[0].message.content or "").strip()
        tf_in  += getattr(rf2.usage, "prompt_tokens", 0)
        tf_out += getattr(rf2.usage, "completion_tokens", 0)
        parsed, err = _parse_json_forecast(tf_text)

    rec.turns.append(Turn(fn, FINAL_TURN_TEMPLATE, "", tf_text, [], [], tf_in, tf_out))

    if parsed:
        rec.structured_forecast = parsed
        rec.yes_prob = float(parsed.get("yes_prob", 0))
    else:
        rec.parse_error = err

    if verbose:
        print(f"     Final done ({tf_out} tokens)  yes_prob={rec.yes_prob}")

    return rec
