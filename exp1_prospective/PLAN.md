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
    ↓
Step 3: Construct counterfactual evidence packets (semi-automated + human review)
    ↓
Step 4: Re-run agent with modified evidence → updated forecasts
    ↓
Step 5: Evaluate evidence–hypothesis and hypothesis–forecast consistency
    ↓
Step 6: Aggregate results + generate report / baselines
```

**Practical ordering**: Run Step 1 once (or reuse cached JSONL from `data/raw_markets/`), then run Step 0 to get the experiment subset, then proceed with Steps 1b–6.

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
│   ├── structured_forecast_agent.py           # Step 2 agent with explicit hypothesis output
│   └── prompts.py                             # System/user prompts for structured output
├── configs/
│   └── exp1_config.yml                        # Run settings
├── data/
│   ├── raw_markets/                           # Step 1 output: full Gamma API dump (JSONL)
│   ├── selected_markets/                      # Step 0 output: filtered experiment subset
│   ├── price_history/                         # Step 1b output: daily price series per market
│   ├── initial_forecasts/                     # Step 2 output: structured forecast JSONL
│   ├── counterfactuals/                       # Step 3 output: counterfactual packets JSONL
│   ├── updated_forecasts/                     # Step 4 output: updated forecast JSONL
│   └── results/                               # Step 5 output: consistency scores + report
└── scripts/
    └── run_full_pipeline.sh                   # One-shot run script
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

For Steps 2–5, the kalshi repo provides:
- `from src.agentic_forecasting.forecast_service import ForecastService, ForecastServiceConfig`
- `from src.agentic_forecasting.polymarket_wiki_agent import PolymarketForecastTask`

### Model / API

- Agent model: `claude-opus-4-8` (via Anthropic API) for best structured output quality
- Counterfactual generation: `claude-sonnet-4-6` (faster, sufficient for packet generation)
- Set `ANTHROPIC_API_KEY` in environment

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

# Agent (Step 2)
model: claude-opus-4-8
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

---

## Recommended Run Order

```bash
# Step 1: pull fresh data (runs in ~30-60s, no API key required)
python fetch_markets/fetch_markets.py

# Step 0: select experiment subset (~1s, runs on cached JSONL)
python fetch_markets/select_markets.py

# Step 1b: fetch price history for selected markets only (~2-5 min at 1s delay)
python fetch_markets/fetch_price_history.py --input data/selected_markets/selected_$(date +%Y-%m-%d).jsonl

# Step 2: initial forecast run (~2-5 min per market, run on ~10 markets for pilot)
python agent/structured_forecast_agent.py --n-markets 10

# Step 3: build counterfactuals (~30s per market, LLM-assisted)
python agent/build_counterfactuals.py

# [Human review of counterfactuals if require_human_review: true]

# Step 4: updated forecasts (~2-5 min per market per counterfactual)
python agent/updated_forecast.py

# Step 5: evaluate and report
python agent/evaluate_consistency.py
```

---

## Open Questions / Design Decisions

1. **How many hypotheses should we require per market?** K=2 is sufficient for initial evaluation, K=3 for richer analysis.

2. **Counterfactual realism vs. testability**: Highly realistic counterfactuals are better for external validity but harder to evaluate automatically. Semi-synthetic counterfactuals allow cleaner direction labeling. Recommend: synthetic for pilot, then upgrade to real evidence modifications.

3. **Should we run the full tool-use loop on counterfactual update, or just a single-turn update?** Start with single-turn append ("given this new info, update"); a full re-search loop is more expensive but cleaner.

4. **Handling hypothesis merging**: If the agent collapses hypotheses when updating, we need a matching step before computing deltas. Use `H_id` from initial output as the key.

5. **Calibration as a sanity check**: Before running the full consistency pipeline, spot-check Step 2 outputs manually for 5–10 markets to confirm the agent produces structured, coherent world models.
