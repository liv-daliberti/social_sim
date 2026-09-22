# Paired context-reversal development summary

Authored, unvalidated development materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates. All three prescribed models are included in fixed order; results must not be used to select a model.

Each model has 20 planned families, 240 context/repeat units, and 960 response records. The design hash is `cf6d5e30b5e3bd3ebd42be5348019c485f2f50f62d5a8b2ae07000ac5cbfb4a4`.

## Coverage and paired outcomes

Direction accuracy and paired reversal use all planned positive/negative trials or pairs. Zero updates and invalid or missing trials fail. Brackets show descriptive 95% family-bootstrap intervals; success/planned counts follow.

| Model | Valid / planned responses | Missing / invalid records | Raw direction | Raw paired reversal | Drift-adjusted direction | Drift-adjusted paired reversal |
|---|---:|---:|---|---|---|---|
| qwen2_5_72b_instruct | 960 / 960 | 0 / 0 | 62.50% [52.48, 73.33]; 75/120 | 30.00% [11.67, 50.00]; 18/60 | 62.50% [52.48, 73.33]; 75/120 | 30.00% [11.67, 50.00]; 18/60 |
| llama3_1_70b_instruct | 960 / 960 | 0 / 0 | 65.00% [53.33, 77.50]; 78/120 | 36.67% [16.67, 56.67]; 22/60 | 65.00% [53.33, 77.50]; 78/120 | 36.67% [16.67, 56.67]; 22/60 |
| qwen3_32b | 960 / 960 | 0 / 0 | 64.17% [55.83, 73.33]; 77/120 | 31.67% [15.00, 48.33]; 19/60 | 64.17% [55.83, 73.33]; 77/120 | 31.67% [15.00, 48.33]; 19/60 |

## Broken-link movement and stability

Movement is in percentage points. Stability requires complete planned measurements, the signed-mean 90% interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance is not equivalence; opposing large revisions cannot establish stability.

| Model | Raw mean absolute [95% interval] | Observed / planned | Raw criterion | Drift-adjusted mean absolute [95% interval] | Observed / planned | Adjusted criterion |
|---|---|---:|---|---|---:|---|
| qwen2_5_72b_instruct | 13.33 [9.17, 17.42] | 60 / 60 | criterion_not_met (complete_planned_observations) | 13.33 [9.17, 17.42] | 60 / 60 | criterion_not_met (complete_planned_observations) |
| llama3_1_70b_instruct | 10.50 [4.75, 17.75] | 60 / 60 | criterion_not_met (complete_planned_observations) | 10.50 [4.75, 17.75] | 60 / 60 | criterion_not_met (complete_planned_observations) |
| qwen3_32b | 13.10 [8.77, 17.52] | 60 / 60 | criterion_not_met (complete_planned_observations) | 13.10 [8.77, 17.52] | 60 / 60 | criterion_not_met (complete_planned_observations) |

## No-news and repeated-news drift

All movements are relative to the shared context-specific baseline, in percentage points. These magnitude summaries use observed complete measurements; missing forecasts are never zero-imputed. Masked contexts have no prespecified null or direction.

| Model | Context | No-news signed mean [95% interval] | No-news mean absolute [95% interval] | Observed / planned | Repeated-news signed mean [95% interval] | Repeated-news mean absolute [95% interval] | Observed / planned |
|---|---|---|---|---:|---|---|---:|
| qwen2_5_72b_instruct | positive | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.87 [0.03, 2.25] | 1.03 [0.03, 2.45] | 60 / 60 |
| qwen2_5_72b_instruct | negative | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.28 [0.00, 0.70] | 0.28 [0.00, 0.70] | 60 / 60 |
| qwen2_5_72b_instruct | broken | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.02 [0.00, 0.05] | 0.02 [0.00, 0.05] | 60 / 60 |
| qwen2_5_72b_instruct | masked | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.17 [0.00, 0.50] | 0.17 [0.00, 0.50] | 60 / 60 |
| llama3_1_70b_instruct | positive | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| llama3_1_70b_instruct | negative | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| llama3_1_70b_instruct | broken | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| llama3_1_70b_instruct | masked | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |
| qwen3_32b | positive | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.33 [0.00, 0.83] | 0.33 [0.00, 0.83] | 60 / 60 |
| qwen3_32b | negative | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 1.67 [0.00, 4.67] | 1.67 [0.00, 4.67] | 60 / 60 |
| qwen3_32b | broken | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.08 [0.00, 0.25] | 0.08 [0.00, 0.25] | 60 / 60 |
| qwen3_32b | masked | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 60 / 60 |

Family resampling preserves all paired contexts, update arms, and repeats. This aggregate retains each model's analysis settings and full metric denominators in JSON. Models appear in the prescribed order, without ranking or selecting a winner.

## Scheduler and run completeness

Plan complete: **true**. See `/n/fs/similarity/social_sim/exp1_prospective/context_reversal/runs/development_v1/completion.json` for accounting and provenance.

| Model | Slurm job | Scheduler state | Complete manifest | Valid / planned records |
|---|---|---|---|---:|
| qwen2_5_72b_instruct | 31440348 | COMPLETED | true | 960 / 960 |
| llama3_1_70b_instruct | 31433799 | COMPLETED | true | 960 / 960 |
| qwen3_32b | 31433855 | COMPLETED | true | 960 / 960 |

Missing or failed model runs remain in the prescribed-model aggregate with all planned denominators. Scheduler completion alone does not establish usable response coverage.
