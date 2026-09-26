# Exp 2: Simulated Sequential Forecasting Worlds — Plan

## What this experiment tests

Models observe a sequence of partially-observable time steps in a synthetic world
where a latent variable governs the dynamics. A model that merely averages past
observations will be miscalibrated; a model that infers the latent state and
reasons about its implications will do strictly better.

This is the "controlled" complement of Exp 1: we *know* the true data-generating
process, so we can measure whether the model has recovered the right inductive
structure, not just whether its forecasts are accurate.

> **Scope (reorganised 2026-06-17).** Exp 2 is now a **single world: Biased News**
> (`biased_news/`), the sequential latent-bias world described in the paper (§4
> "Election World"). The former "World 1 — Elections" (the *static* 28-node DAG +
> GRPO training UI) was **not sequential** and its training pipeline is Exp-3
> material, so it moved to `exp3_training_transfer/elections/`. The candidate
> "World 3+" worlds below remain Exp-3 transfer targets, not Exp-2 environments.

---

## The world — Biased News (`biased_news/`)

**Status:** engine + canonical eval pipeline built; 2/10 baseline model runs done
(llama3.1-8b, qwen2.5-7b). No frontier runs launched yet.

### Motivation

This is the world described in the paper (Section 3, Figure 1). The latent
variable is the *news bias* of the city: in a biased city, negative events have
only a minor effect on public opinion; in a neutral city, the same events produce
larger shifts. The model never directly observes bias or opinion — only events and
survey results.

A model that simply averages past event-survey associations will underestimate
variance in biased cities and overestimate it in neutral ones. A model that
maintains a hypothesis over the latent variable (biased vs neutral) and weights
its forecast accordingly will do better.

### Causal structure (per city, per episode)

```
Bias  ─────────────────────────────────────┐   (root, never observed)
                                            │
Events_t ──→ ΔOpinion_t ←────────── Bias  │   (via effect_size)
               │                           │
             Opinion_t = Opinion_{t-1} + ΔOpinion_t + noise
               │
             Survey_t ~ Normal(Opinion_t, σ_survey)   (observed)
```

The model sees: `(Events_1, Survey_1), (Events_2, Survey_2), ..., (Events_T, Survey_T)`
and predicts `Survey_{T+1}`.

### Parameters

```
opinion_init   ~ Uniform(40, 60)         # starting opinion (scale 0–100)
bias           ~ Bernoulli(0.5)           # latent: "biased" or "neutral"

# Effect of a negative event on opinion
effect_negative_neutral = -8.0           # full effect in neutral city
effect_negative_biased  = -2.0           # dampened in biased city

# Positive events (mirror symmetry)
effect_positive_neutral = +6.0
effect_positive_biased  = +1.5

# Neutral events: no effect
effect_neutral = 0.0

# Opinion noise per step
σ_opinion  = 2.0                         # random walk noise on opinion
σ_survey   = 4.0                         # measurement noise on survey

# Sequence length
T ~ Uniform_int(6, 12)                   # steps per episode (T observed, predict T+1)
```

Opinion is clipped to [0, 100]. Survey rounds to integer for the prompt.

### Event vocabulary

Events are sampled i.i.d. from:

| Event type          | P(event) | Description shown in prompt                   |
|---------------------|----------|-----------------------------------------------|
| negative_scandal    | 0.20     | "A corruption scandal involving the candidate was reported." |
| negative_gaffe      | 0.15     | "The candidate made a significant gaffe in a major speech." |
| negative_poll_drop  | 0.10     | "Internal party polling showed unexpected weakness." |
| positive_endorsement| 0.20     | "A major national figure endorsed the candidate." |
| positive_rally      | 0.15     | "A large and enthusiastic rally was held in a key district." |
| neutral_debate      | 0.10     | "The candidates debated without a clear winner." |
| neutral_nothing     | 0.10     | "No major campaign events occurred."          |

Event polarity (positive/negative/neutral) determines the opinion effect;
the specific event text is the only thing shown to the model.

### Context variations (the "polling company" axis)

Three prompt variants, crossed with all tasks:

| Variant | Context prefix added to prompt                                              |
|---------|-----------------------------------------------------------------------------|
| V0      | (no mention of polling company)                                             |
| V1      | "Surveys are conducted by CityView Polling. Note: one major polling firm in this region is known to have a systematic partisan lean." |
| V2      | "Surveys are conducted by Hartfield Analytics, which is owned by a close relative of one of the candidates. This firm has faced past scrutiny for biased reporting." |

V1 provides a weak signal that a biased pollster *exists* (but doesn't identify it).
V2 directly flags the specific pollster as potentially biased.

The question: does V2 > V1 > V0 in calibration, and does the model *use* this
information to infer latent bias before it's statistically evident from the data?

### Task format (jsonl)

```json
{
  "task_id": "city_00042_v1",
  "variant": "V1",
  "context_prefix": "Surveys are conducted by CityView Polling...",
  "history": [
    {"step": 1, "event": "A corruption scandal involving the candidate was reported.", "survey": 52},
    {"step": 2, "event": "A major national figure endorsed the candidate.",            "survey": 53},
    ...
  ],
  "question": "Based on the events and surveys above, what survey result (0–100) do you expect in the next period?",
  "settlement_value": 49.3,      // true E[Survey_{T+1}]
  "settlement_rounded": 49,      // rounded integer (for human readability)
  "_hidden_bias": "biased",
  "_hidden_opinion_trajectory": [55.1, 53.4, ...],
  "_hidden_true_opinion_at_T1": 49.3
}
```

Reward: negative absolute error (or negative squared error) against `settlement_value`.

### Dataset sizes

| Split         | Cities | Variants | Total tasks |
|---------------|--------|----------|-------------|
| train         | 2000   | V0+V1+V2 | 6000        |
| eval_in_dist  | 500    | V0+V1+V2 | 1500        |
| eval_out_dist | 200    | V0 only  | 200         |

Out-of-dist eval: sequences drawn with T=16–20 (longer than any training sequence).

### Metrics

**Primary:**
- MAE (mean absolute error) against true E[Survey_{T+1}]
- Calibration: expected vs actual error by decile

**Latent inference probes (analysis, not training signal):**
- After the model forecasts, ask it: "Do you think the polling in this city is biased (yes/no)?"
  Compare accuracy by ground-truth bias label.
- Error gap: |MAE_biased − MAE_neutral|. A model that has inferred the latent
  should have low gap (corrects for the dampening); a naive model will show
  large gap (systematically over-reacts in biased cities).
- Context sensitivity: does V2 reduce the error gap vs V0? Does it do so from
  step 1 (prior injection) or only after several steps (Bayesian updating)?

### Layout (current, mirrors `exp1_prospective/`)

```
exp2_simulated_worlds/biased_news/
├── engine/temporal_dag.py       # ✅ DGP + compute_bayes_forecast (Kalman oracle) + naive EMA
├── eval/                        # ✅ CANONICAL eval pipeline (Pipeline A)
│   ├── run_batch.py             #    per-T-prefix records; --backend ollama|azure; writes data/batch/results_*.jsonl
│   ├── slurm_eval.sh            #    param'd SLURM (MODEL env) — supersedes slurm_qwen/llama.sh
│   ├── launch_local.sh          #    7 local Ollama models
│   ├── launch_frontier.sh       #    3 Azure frontier models
│   └── _archive/                #    slurm_qwen.sh, slurm_llama.sh (superseded)
├── configs/biased_news_v1.yml
├── data/batch/                  # results_{model}_{date}.jsonl  (+ .manifest.json)
├── tests/test_engine.py         # ✅ 8 passing
├── viewer/                      # ✅ single app: Simulator + Results screens
│   ├── app.py                   #    port 5052; reads data/batch/
│   └── templates/index.html
├── _archive/                    # redundant standalone simulator app (server_standalone.py, ui_simulator_index.html)
└── scripts/_archive/            # GRPO training cluster — Exp-3 material, parked here
    ├── run_grpo.py              #    GRPO training + (dead) eval mode
    ├── run_{claude,gpt,deepseek,llama,qwen}_eval.py   # dead Pipeline-B wrappers
    └── generate_tasks.py        #    task-file generator for the GRPO pipeline
```

### Remaining steps (P1–P4)

1. [x] Engine, canonical eval pipeline, single viewer — **done**
2. [ ] **P1:** launch 10-model evals (`eval/launch_local.sh` + `eval/launch_frontier.sh`)
3. [ ] **P2:** `analyze.py` report — MAE, excess-MAE-vs-oracle, error-gap |MAE_biased−MAE_neutral|, V0/V1/V2 by step T, latent-bias probe
4. [ ] **P3:** Exp 2 figure (mirror `paper/generate_figures.py`)
5. [ ] **P4:** `experiment2_section.tex` + `experiment2_appendix.tex` (mirror Exp 1 depth)

> **Note on training.** GRPO is reserved for Exp 3, so the training code (`run_grpo.py`,
> `generate_tasks.py`) lives in `scripts/_archive/` — recoverable, but out of the
> Exp-2 (eval-only) path. The canonical Exp-2 eval is `eval/run_batch.py`.

---

## World 3+ — Additional worlds (for Exp 3 transfer)

The paper tests whether training on a *family* of synthetic worlds transfers to
held-out worlds. We need at least 2–3 more worlds with structurally similar but
surface-distinct DGPs so the held-out-world test is meaningful.

### Candidate World 3: Economic Indicators (`economic_trend/`)

**Latent:** Regime ∈ {expansion, contraction} — drawn once per episode.
**Observed per step:** A noisy economic indicator (e.g., "employment report" on
a 0–100 scale).
**Events:** Policy decisions, trade announcements, central bank decisions.
**Effect:** In expansion, positive policy events boost the indicator by more;
in contraction, negative events drag it down more.
**Task:** Predict the next indicator reading.
**Variation:** The "reporting agency" is either government (neutral) or industry
lobby (potentially biased upward).

### Candidate World 4: Drug Trial (`clinical_trial/`)

**Latent:** Drug efficacy ∈ {effective, ineffective} — drawn once per trial arm.
**Observed per step:** Weekly response rates (0–100%) from a clinical cohort.
**Events:** Protocol changes, adverse events, dropouts.
**Effect:** In an effective arm, adverse events produce smaller permanent response
drops (the drug "recovers" patients faster); in an ineffective arm, response
tracks noise.
**Task:** Predict next week's response rate.
**Variation:** Context says nothing / context notes the trial is sponsored by
the drug manufacturer / context says a regulatory body is independently monitoring.

### Candidate World 5: Social Media Trend (`social_trend/`)

**Latent:** Organic vs manufactured virality.
**Observed per step:** Daily share counts for a piece of content.
**Events:** Celebrity mentions, counter-narratives, platform algorithm changes.
**Effect:** Manufactured virality dampens the effect of counter-narratives
(bot accounts keep sharing regardless); organic virality responds more to them.
**Task:** Predict next day's share count.

---

## Experiment 3 mapping

| Train worlds             | Held-out world         | Real-world transfer       |
|--------------------------|------------------------|---------------------------|
| biased_news + economic   | clinical_trial         | Polymarket pastcasting     |
| biased_news + clinical   | economic_trend         | Polymarket pastcasting     |
| All 4 worlds             | social_trend           | Polymarket pastcasting     |

Training = GRPO on a mix of worlds.
Transfer = zero-shot or few-shot performance on the held-out world.
Null hypothesis = train only on the held-out world itself.

---

## Open questions

1. **Continuous vs categorical survey:** Should Survey_t be a continuous float or
   a discretized integer (e.g., 0–100)? Integer is more natural for prompts;
   float gives a cleaner reward signal. Recommendation: integer in the prompt,
   float as the reward target.

2. **Sequence length vs. difficulty:** Short sequences (T=4) give little evidence
   about the latent; long sequences (T=12) make it identifiable. Should we
   control T during training or let it vary? Recommend: vary T uniformly and
   report metrics stratified by T.

3. **Reward shape for the temporal task:** MSE rewards the expected value but not
   uncertainty. Should we ask the model for a distribution (e.g., "give a 90%
   interval") and use log-score? This would more directly test calibration.
   Start with MAE for simplicity; add interval scoring in v2.

4. **Confound between V1/V2 and statistical evidence:** V2 provides a strong
   prior even at step 1. This conflates prior injection with Bayesian updating.
   Consider a "V2 late" condition where the company information is revealed
   only after step T/2 — this isolates the updating effect.
