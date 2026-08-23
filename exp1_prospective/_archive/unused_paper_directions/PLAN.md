# Experiment 1: Prospective Forecasting and World-Model Consistency
## Step-by-Step Implementation Plan

**Goal**: Test whether frontier LLM agents construct coherent, updateable event-specific world models when forecasting unresolved Polymarket markets — or instead rely on shortcuts, post-hoc rationalization, or market imitation.

**Key question**: When the agent's stated hypotheses and evidence change (via counterfactual interventions), do the forecast probabilities update consistently with what the agent's own model predicts they should?

---

## Pipeline Overview

```
Step 0: Select the experiment subset from raw Gamma API data
    ↓
Step 1: (One-time) Fetch ALL active markets from Polymarket Gamma API
    ↓  (feeds Step 0)
Step 1b: Fetch price history for selected markets only (CLOB API)
    ↓
Step 2: Run structured world-model agent → initial forecast + explicit hypothesis structure
    │
    ├── 2a: Remote frontier agents (GPT-5.4 via Azure, Claude Opus via Azure)
    │         [web search handled by Azure AI Foundry agent_reference]
    │
    └── 2b: Local small models (Qwen2.5-7B, Llama3.1-8B via Ollama)
              [web search via explicit tool-calling loop + Tavily/DuckDuckGo]
    ↓
Step 3: Construct counterfactual evidence packets (semi-automated + human review)
    ↓
Step 4: Re-run agent with modified evidence → updated forecasts (all model variants)
    ↓
Step 5: Evaluate evidence–hypothesis and hypothesis–forecast consistency
    ↓
Step 6: Aggregate results + generate report / baselines
```

**Practical ordering**: Run Step 1 once (or reuse cached JSONL from `data/raw_markets/`), then run Step 0 to get the experiment subset, then proceed with Steps 1b–6.

**Model comparison structure**: Steps 2b–5 are run independently for each model variant, producing parallel forecast JSONL files with identical schema. Step 6 aggregates across all variants for cross-model comparison.

---

## File Structure

```
exp1_prospective/
├── PLAN.md                                    ← this file
├── fetch_markets/                             # self-contained data-collection module
│   ├── gamma_api.py                           #   vendored Gamma API client
│   ├── price_history_api.py                   #   vendored CLOB price-history client
│   ├── fetch_markets.py                       #   Step 1: fetch all active markets
│   ├── select_markets.py                      #   Step 0: filter/select experiment subset
│   └── fetch_price_history.py                 #   Step 1b: CLOB price history for selected
├── agent/
│   ├── forecast_agent.py                      # Step 2a: Azure Foundry agent (GPT / Claude)
│   ├── run_forecast.py                        #   run script for GPT agent
│   ├── run_claude_forecast.py                 #   thin wrapper for Claude Opus variant
│   ├── local_agent.py                         # Step 2b: local model agent (Qwen / Llama) [NEW]
│   ├── local_tools.py                         #   web search tool: Tavily + DDG fallback [NEW]
│   ├── run_local_forecast.py                  #   run script for local model variants   [NEW]
│   ├── build_counterfactuals.py               # Step 3: counterfactual packet builder
│   ├── updated_forecast.py                    # Step 4: GPT counterfactual update
│   ├── claude_updated_forecast.py             #   Claude variant of Step 4
│   ├── local_updated_forecast.py              #   local model variant of Step 4         [NEW]
│   ├── evaluate_consistency.py                # Step 5: consistency metrics
│   └── prompts.py                             # Shared prompt templates (all model variants)
├── configs/
│   ├── exp1_config.yml                        # Run settings (frontier models)
│   └── local_models_config.yml                # Local model run settings                [NEW]
├── data/
│   ├── raw_markets/                           # Step 1 output: full Gamma API dump (JSONL)
│   ├── selected_markets/                      # Step 0 output: filtered experiment subset
│   ├── price_history/                         # Step 1b output: daily price series per market
│   ├── initial_forecasts/                     # Step 2 output: forecasts_{model}_{date}.jsonl
│   ├── counterfactuals/                       # Step 3 output: counterfactual packets JSONL
│   ├── updated_forecasts/                     # Step 4 output: updated_{model}_{date}.jsonl
│   └── results/                               # Step 5 output: consistency scores + report
└── scripts/
    ├── run_full_pipeline.sh                   # One-shot run script (frontier models)
    └── run_local_pipeline.sh                  # One-shot run script (local models)       [NEW]
```

---

## Step 0: Select the Experiment Subset

**Script**: `fetch_markets/select_markets.py`

**What it does**: Applies filters to the raw Gamma API dump (`data/raw_markets/`) to produce a curated ~50–100 market subset that is (a) substantively interesting for world-model evaluation, (b) not trivially predictable, and (c) has enough information for a structured agent to reason about.

This is Step 0 because defining *what we care about* is the first decision — the fetch (Step 1) is a prerequisite data-collection operation that only needs to run once.

**Input**: `data/raw_markets/markets_{YYYY-MM-DD}.jsonl`
**Output**: `data/selected_markets/selected_{YYYY-MM-DD}.jsonl`

**Filters applied** (in order):

1. **Binary only**: keep markets where `outcomes == ["Yes", "No"]` (or equivalent). Multivariate markets complicate consistency scoring.

2. **Active and unresolved**: `active=True`, `closed=False`, `resolved=False`.

3. **Category / topic filter**: keep markets tagged with Politics, Government, Elections, Economics, Geopolitics, or Science — OR where the question/event text matches relevant keywords. Crypto/sports/entertainment markets are excluded.

4. **Resolution window**: keep markets resolving between **14 and 180 days** from fetch date. Too short = nearly certain; too long = no grounding signal for counterfactual testing.

5. **Non-trivial price**: keep markets where `0.10 < yes_price < 0.90`. Avoids markets already near certainty.

6. **Minimum volume**: keep markets with `volume_usd >= 1000`. Thin markets may not be well-formed questions.

7. **Deduplication**: within the same underlying event (`event_id`), keep at most the one market with the highest volume. Avoids asking the agent nearly-identical questions.

**Output record** (same fields as Step 1 plus):
```json
{
  "...",
  "task_id": "pm_{market_id}_{fetch_date}",
  "days_to_resolution": 42,
  "selected_at": "2026-06-05T..."
}
```

**Target size**: ~50–100 markets for initial run.

---

## Step 1: Fetch Active Markets from Polymarket Gamma API

**Script**: `fetch_markets/fetch_markets.py`

**What it does**: Paginates the Polymarket Gamma API to exhaustion, retrieves all currently active, unresolved markets, and writes one JSON record per market to a JSONL file.

**Key implementation details**:
- Keyset pagination (`/events/keyset`) with fallback to offset pagination
- Extracts all volume windows (24h, 1wk, 1mo, 1yr), price changes, CLOB token IDs
- Live terminal counter while fetching (pages, events, markets, active+binary)
- Output: `data/raw_markets/markets_{YYYY-MM-DD}.jsonl` + manifest

**Run once** (or reuse an existing JSONL if it's recent). The full dump (~61k markets) takes ~30–60 seconds.

---

## Step 1b: Fetch Price History (CLOB API)

**Script**: `fetch_markets/fetch_price_history.py`

**What it does**: For each selected market, fetches its daily Yes-token price series from the Polymarket CLOB API.

**Key implementation details**:
- Must run on filtered subset (~100 markets), NOT the full 61k dump
- Serial requests with configurable delay (default 1.0s) — CLOB is behind Cloudflare
- Chunks long-lived markets into 28-day windows to avoid `"interval too long"` errors
- Output: `data/price_history/price_history_{YYYY-MM-DD}.jsonl`

---

## Step 2: Run Structured World-Model Agent (Initial Forecast)

**Script**: `agent/structured_forecast_agent.py`

**What it does**: This is the most critical step. For each selected market, we run a frontier LLM agent that produces not just a forecast probability, but an explicit structured world model: key actors, mechanisms, hypotheses with probabilities, and a mapping from hypothesis probabilities to the final forecast.

### Why a new agent (not just `ForecastService`)

The existing `forecast_with_wikipedia_tools()` is optimized for a single `yes_prob` output. For Exp 1, we need the agent to explicitly articulate:
- A set of **K competing hypotheses** (H1…HK) that cover the key uncertainty
- A **probability for each hypothesis**
- A **mapping** from hypotheses to the outcome (i.e., "if H1 holds, P(YES) = 0.8; if H2 holds, P(YES) = 0.2")
- The **evidence** it used for each hypothesis assessment
- A **final forecast** derived from those hypotheses

This enables us to separately test Evidence→Hypothesis consistency and Hypothesis→Forecast consistency.

### New agent design

Wraps existing `ForecastService` / `forecast_with_wikipedia_tools()` infrastructure with a modified system prompt that demands structured JSON output. Key changes:

**Structured output format** (see `agent/prompts.py`):

```
{
  "event_model": {
    "key_actors": [...],
    "key_mechanisms": [...],
    "latent_variables": [...]
  },
  "hypotheses": [
    {
      "id": "H1",
      "description": "...",
      "supporting_evidence": ["...", "..."],
      "contradicting_evidence": ["...", "..."],
      "probability": 0.65
    },
    ...
  ],
  "hypothesis_to_forecast_mapping": "If H1 holds (p=0.65): P(YES)=0.8. If H2 holds (p=0.35): P(YES)=0.15. Combined: 0.65*0.8 + 0.35*0.15 = 0.57",
  "yes_prob": 0.57,
  "rationale": "..."
}
```

**Output**: `data/initial_forecasts/forecasts_{YYYY-MM-DD}.jsonl`

---

## Step 2b: Local Model Agent (Qwen2.5-7B / Llama3.1-8B)

**Scripts**: `agent/local_agent.py`, `agent/local_tools.py`, `agent/run_local_forecast.py`

**Scientific purpose**: Demonstrate that the same world-model construction and consistency behavior observed in frontier models also appears in 7–8B open-weight models running locally. This strengthens the generality of the finding and rules out "behavior emerges only at scale" as an alternative explanation.

### Why this requires a different architecture

The frontier agents (Step 2a) use Azure AI Foundry's `agent_reference` protocol, which delegates web search to Azure's hosted agent infrastructure — the model calls search implicitly and Azure handles tool execution. Local models via Ollama expose a plain OpenAI-compatible chat API with no such delegation layer. Web search must be implemented as an explicit function-calling tool that the model invokes in a ReAct-style loop.

### Serving infrastructure

Use **Ollama** (`https://ollama.com`) as the local serving layer:

```bash
# Install and pull models (one-time setup)
ollama pull qwen2.5:7b          # Qwen2.5-7B-Instruct (32k context)
ollama pull llama3.1:8b          # Llama3.1-8B-Instruct (128k context)

# Ollama exposes OpenAI-compatible API at:
#   http://localhost:11434/v1
```

Both models support OpenAI-compatible function/tool calling, which Ollama exposes correctly via its `/v1/chat/completions` endpoint.

### Web search tooling (`local_tools.py`)

Define a single `web_search` tool with two backends:

1. **Tavily** (primary, `tavily-python` SDK): designed for AI agent use, returns clean snippet-level results. Free tier: 1000 searches/month. Set `TAVILY_API_KEY` in environment.
2. **DuckDuckGo** (fallback, `duckduckgo-search` package): free, no API key, rate-limited. Use if Tavily key is absent or exhausted.

Tool schema (passed to model as `tools=[...]`):
```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for current information about a topic.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query string"},
        "max_results": {"type": "integer", "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

Search results are truncated to `max_chars_per_result` (default 800) and injected as a tool-role message before the next model call.

### Agent loop (`local_agent.py`)

Implements a `LocalForecastAgent` class following the same multi-turn flow as `forecast_agent.py`:

```
Turn 1 (no tools): event model + initial hypothesis framing
    [same TURN1_TEMPLATE from prompts.py]

Tool loop (web_search enabled):
    while model emits tool_calls:
        - execute each web_search call via local_tools.py
        - append tool_role messages with results
        - call model again
    → model eventually produces a non-tool-call response (T2 narrative)
    [TURN2_TEMPLATE content is the initial user message that starts this loop]

Turn 3 (optional, tool loop): deepen on weakest evidence
    [same TURN3_TEMPLATE; re-enters tool loop]

Final turn (no tools, JSON mode if supported):
    [same FINAL_TURN_TEMPLATE; forces structured JSON output]
    → if model output is not valid JSON, attempt one repair prompt
```

Key differences from frontier agents:
- No `previous_response_id` threading — each market is a single multi-turn chat list
- Tool loop is explicit and bounded by `max_search_calls` (default 6)
- `response_format={"type": "json_object"}` passed on final turn (Ollama supports this for both models)
- JSON repair fallback: if final output fails to parse, send one follow-up asking the model to re-emit only the JSON block

### Run script (`run_local_forecast.py`)

Same interface as `run_forecast.py`:

```bash
# Run Qwen2.5-7B on first 10 markets, 3 runs each
python agent/run_local_forecast.py --model qwen2.5:7b --n 10 --k 3

# Run Llama3.1-8B, verbose, no third turn
python agent/run_local_forecast.py --model llama3.1:8b --no-third-turn --verbose

# Dry run (print prompts, no API calls)
python agent/run_local_forecast.py --model qwen2.5:7b --dry-run
```

Flags:
- `--model`: Ollama model tag (default: `qwen2.5:7b`)
- `--endpoint`: Ollama base URL (default: `http://localhost:11434/v1`)
- `--max-search-calls`: cap on web_search calls per turn loop (default: 6)
- `--no-third-turn`: skip Turn 3 (faster, ~2 fewer search rounds)
- `--search-backend`: `tavily` | `duckduckgo` | `auto` (default: `auto`)
- All other flags match `run_forecast.py`: `--input`, `--out`, `--n`, `--k`, `--delay`, `--verbose`, `--dry-run`

**Output naming**: `data/initial_forecasts/forecasts_{model-slug}_{YYYY-MM-DD}.jsonl`
- e.g., `forecasts_qwen2.5-7b_2026-06-09.jsonl`, `forecasts_llama3.1-8b_2026-06-09.jsonl`

**Output schema**: Identical to frontier agent output. Adds one field: `"backend": "local-ollama"`. Tool calls from the explicit loop are recorded in `turns[*].tool_calls` exactly as Azure tool calls are, enabling direct comparison of search behavior across model variants.

### Step 4 variant (`local_updated_forecast.py`)

Counterfactual injection for local models follows the same single-turn injection as `updated_forecast.py` (no tools on the update step — just "here is new evidence, update your JSON"). This is simpler and keeps the counterfactual comparison clean:

```bash
python agent/local_updated_forecast.py --model qwen2.5:7b
python agent/local_updated_forecast.py --model llama3.1:8b
```

**Output**: `data/updated_forecasts/updated_{model-slug}_{YYYY-MM-DD}.jsonl`

---

## Step 3: Construct Counterfactual Evidence Packets

**Script**: `agent/build_counterfactuals.py`

**What it does**: For each market, constructs 2–3 counterfactual evidence snippets that alter the evidence in directionally interpretable ways. Each packet has a known expected direction of effect on one of the agent's stated hypotheses.

### Counterfactual types

1. **Hypothesis-strengthening packet**: increases P(H1). Label: `direction="pro_H1"`.
2. **Hypothesis-weakening packet**: decreases P(H1). Label: `direction="anti_H1"`.
3. **Orthogonal packet** (optional): evidence relevant to H2 but not H1.

**Generation**: LLM-assisted (claude-sonnet-4-6) + human review pass to verify coherence and directional unambiguity.

**Output**: `data/counterfactuals/counterfactuals_{YYYY-MM-DD}.jsonl`

---

## Step 4: Re-Run Agent with Counterfactual Evidence

**Script**: `agent/updated_forecast.py`

**What it does**: For each valid counterfactual packet, re-runs the structured world-model agent with the counterfactual evidence injected as a "new finding."

**Injection mode (initial implementation — Mode A)**:
```
NEW EVIDENCE (received after your initial research):
---
{evidence_text}
---
Given this new evidence, update your structured world model. Produce a new JSON with the same format,
showing how your hypothesis probabilities and final forecast change.
```

**Output**: `data/updated_forecasts/updated_{YYYY-MM-DD}.jsonl`

---

## Step 5: Evaluate Consistency

**Script**: `agent/evaluate_consistency.py`

**What it does**: Computes three consistency metrics for each (market, counterfactual) pair.

### Metric 1: Evidence–Hypothesis Consistency (EHC)

```
EHC(cf) = 1  if sign(Δhypothesis_prob_k) == expected_direction
         = 0  otherwise
         = NaN if |Δhypothesis_prob_k| < 0.03 (no meaningful update)
```

### Metric 2: Hypothesis–Forecast Consistency (HFC)

```python
delta_implied = sum(delta_h_k * mapping_weight_k for k, delta_h_k in hypothesis_deltas.items())
delta_actual  = updated_yes_prob - initial_yes_prob
HFC(cf) = 1 if sign(delta_actual) == sign(delta_implied) else 0
```

### Metric 3: Internal Coherence Score (ICS)

Check whether `Σ P(H_k) * P(YES|H_k) ≈ updated_yes_prob` still holds after updating.

### Aggregate report

Per market: `EHC_rate`, `HFC_rate`, `ICS_rate`
Across all markets: mean ± SE of each rate, breakdown by counterfactual type and question category.

**Output**: `data/results/consistency_report_{YYYY-MM-DD}.json` + `data/results/summary_{YYYY-MM-DD}.md`

---

## Step 6: Baselines and Controls

Run in parallel with the main pipeline:

1. **Anchoring check**: Does `updated_yes_prob` simply track the initial `yes_prob` regardless of counterfactual direction?

2. **Market-price baseline**: Compare `initial_yes_prob` to Polymarket `last_price_yes` at fetch time. Measures whether the agent reproduces market prices vs. independent reasoning.

3. **No-evidence control**: Run the same structured agent prompt with no evidence-gathering tools. Measures how much of the initial forecast comes from parametric knowledge vs. retrieved evidence.

---

## Technical Dependencies

### Python environment

Key imports (from the self-contained `fetch_markets/` module):
- `from gamma_api import fetch_events_keyset_page, fetch_events_page` — Step 1
- `from price_history_api import fetch_price_history` — Step 1b

For Steps 2a (frontier agents):
- `openai` — OpenAI-compatible client for Azure AI Foundry

For Steps 2b (local agents), additional packages:
- `openai` — reused; Ollama exposes the same `/v1/chat/completions` API
- `tavily-python` — Tavily web search SDK (`pip install tavily-python`)
- `duckduckgo-search` — free fallback search (`pip install duckduckgo-search`)
- `ollama` — optional management client (`pip install ollama`)

### Model / API

**Frontier agents (Step 2a)**:
- GPT-5.4 via Azure AI Foundry (`AZURE_AI_API_KEY`, `AZURE_PROJECT_ENDPOINT`)
- Claude Opus 4-8 via Azure AI Foundry (`CLAUDE_AZURE_API_KEY`)
- Counterfactual generation: `claude-sonnet-4-6` (faster, sufficient for packet generation)

**Local agents (Step 2b)**:
- Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct via Ollama (no API key needed)
- Tavily web search: set `TAVILY_API_KEY` (free tier: 1000 searches/month)
- DuckDuckGo search: no key required (automatic fallback if Tavily unavailable)

### Config file: `configs/exp1_config.yml`

```yaml
# Selection filters (Step 0)
min_days_to_resolution: 14
max_days_to_resolution: 180
min_yes_price: 0.10
max_yes_price: 0.90
min_volume_usd: 1000
categories: [Politics, Government, Elections, Economics, Geopolitics, Science]
target_sample_size: 75

# Data pull (Step 1)
fetch_date: auto          # YYYY-MM-DD, defaults to today
page_limit: 100

# Frontier agent (Step 2a)
model: gpt-5.4
max_tool_rounds: 3
min_tool_rounds: 1
max_tokens: 1500
temperature: 0.1

# Counterfactuals (Step 3)
counterfactuals_per_market: 2    # pro + anti for top hypothesis
cf_generation_model: claude-sonnet-4-6
require_human_review: true

# Evaluation (Step 5)
consistency_min_delta: 0.03      # min hypothesis prob shift to count as "updated"
```

### Config file: `configs/local_models_config.yml`

```yaml
# Local model serving
ollama_endpoint: http://localhost:11434/v1

# Web search
web_search:
  provider: auto              # "tavily" | "duckduckgo" | "auto" (uses Tavily if key set)
  max_results: 5
  max_chars_per_result: 800   # truncate each snippet to keep context manageable

# Models to run (processed sequentially by run_local_forecast.py)
models:
  - model_id: qwen2.5:7b
    display_name: qwen2.5-7b
    max_tokens: 1500
    temperature: 0.1
    max_search_calls: 6       # cap on web_search calls per turn-loop pass
    do_third_turn: true

  - model_id: llama3.1:8b
    display_name: llama3.1-8b
    max_tokens: 1500
    temperature: 0.1
    max_search_calls: 6
    do_third_turn: true

# JSON repair: if final turn output fails to parse, send one repair prompt
json_repair_attempts: 1

# Output
k_runs: 3                    # runs per market (matches frontier agent default)
```

---

## Recommended Run Order

```bash
# ── One-time setup ────────────────────────────────────────────────────────────

# Install Ollama and pull small models (local only, ~5-10 min download)
ollama pull qwen2.5:7b
ollama pull llama3.1:8b

# Install extra Python deps for local agent
pip install tavily-python duckduckgo-search

# Set search key (optional; DDG fallback works without it)
export TAVILY_API_KEY=<your-key>


# ── Data collection (shared across all model variants) ─────────────────────────

# Step 1: pull fresh markets (~30-60s, no API key required)
python fetch_markets/fetch_markets.py

# Step 0: select experiment subset (~1s)
python fetch_markets/select_markets.py

# Step 1b: price history for selected markets (~2-5 min)
python fetch_markets/fetch_price_history.py --input data/selected_markets/selected_$(date +%Y-%m-%d).jsonl


# ── Step 2a: Frontier agents ──────────────────────────────────────────────────

# GPT-5.4 via Azure (requires AZURE_AI_API_KEY)
python agent/run_forecast.py --n 10 --k 3

# Claude Opus 4-8 via Azure (requires CLAUDE_AZURE_API_KEY)
python agent/run_claude_forecast.py --n 10 --k 3


# ── Step 2b: Local small models ───────────────────────────────────────────────

# Make sure Ollama is running: `ollama serve` (or it starts automatically)

# Qwen2.5-7B (~3-8 min per market depending on hardware; pilot with n=5)
python agent/run_local_forecast.py --model qwen2.5:7b --n 5 --k 3 --verbose

# Llama3.1-8B
python agent/run_local_forecast.py --model llama3.1:8b --n 5 --k 3 --verbose

# Dry run to check prompts without API calls
python agent/run_local_forecast.py --model qwen2.5:7b --dry-run


# ── Step 3: Counterfactuals (model-agnostic; build once) ──────────────────────

python agent/build_counterfactuals.py
# [Human review if require_human_review: true]


# ── Step 4: Updated forecasts (run per model variant) ─────────────────────────

python agent/updated_forecast.py                       # GPT
python agent/claude_updated_forecast.py                # Claude
python agent/local_updated_forecast.py --model qwen2.5:7b
python agent/local_updated_forecast.py --model llama3.1:8b


# ── Step 5: Evaluate consistency ─────────────────────────────────────────────

# Runs on all model variants by default; use --model to scope
python agent/evaluate_consistency.py
```

---

## Open Questions / Design Decisions

1. **How many hypotheses should we require per market?** K=2 is sufficient for initial evaluation, K=3 for richer analysis.

2. **Counterfactual realism vs. testability**: Highly realistic counterfactuals are better for external validity but harder to evaluate automatically. Semi-synthetic counterfactuals allow cleaner direction labeling. Recommend: synthetic for pilot, then upgrade to real evidence modifications.

3. **Should we run the full tool-use loop on counterfactual update, or just a single-turn update?** Start with single-turn append ("given this new info, update"); a full re-search loop is more expensive but cleaner. Decision currently: single-turn for all model variants for comparability.

4. **Handling hypothesis merging**: If the agent collapses hypotheses when updating, we need a matching step before computing deltas. Use `H_id` from initial output as the key.

5. **Calibration as a sanity check**: Before running the full consistency pipeline, spot-check Step 2 outputs manually for 5–10 markets to confirm the agent produces structured, coherent world models.

### Local model-specific questions

6. **JSON output reliability at 7-8B scale**: Small models are less reliable at producing valid structured JSON. Three mitigations in order: (a) `response_format={"type": "json_object"}` on final turn (Ollama supports this), (b) one repair prompt if parse fails, (c) flag as `parse_error` and exclude from analysis. Track parse failure rate per model — if >20%, the model variant may not be viable.

7. **Search quality at small scale**: Frontier agents issue ~3–6 targeted web searches per market. Small models may issue fewer, more generic, or redundant queries. Log all `tool_calls` to measure search behavior differences across model variants.

8. **Hardware requirements**: Qwen2.5-7B and Llama3.1-8B both fit in ~6–8 GB VRAM (4-bit quant) or ~14–16 GB system RAM via Ollama's CPU fallback. At CPU speed expect ~30–90 min per market for the full tool loop. Use `--no-third-turn` for pilots on slow hardware.

9. **Comparability of search access**: Azure agents search via Bing (via Azure Cognitive Search); local agents search via Tavily or DuckDuckGo. This is a potential confound. Mitigations: (a) note in paper that search backend differs, (b) run one local-agent condition with search disabled to isolate parametric vs. retrieved knowledge contribution.

10. **Scale of local-model arm**: Full 75-market × 3-run × 2-models run is ~450 inference calls + 900 web searches. For a paper showing generalization, n=20–30 markets × 1–2 runs is sufficient for significance; pilot with 10 markets first.
