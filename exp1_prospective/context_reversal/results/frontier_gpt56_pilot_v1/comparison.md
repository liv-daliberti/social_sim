# Matched repeat-0 frontier development comparison

Exploratory deployment comparison on authored, independently unvalidated development materials. All 20 original families and all four contexts are retained, using repeat 0 only. The frontier extension was added after local outcomes. Generation settings differ: local temperature 0.7 with constrained decoding and 128 output tokens; GPT-5.6 uses low reasoning with 2048 output tokens and temperature/seed omitted. Qwen3 uses non-thinking mode. These descriptive results do not isolate model capability, establish a confirmatory ranking, or support population claims or model selection.

Each model has 320/320 valid responses: 20 families × four contexts × one baseline and three update arms. Correct sign uses 40 positive/negative trials; paired reversal uses 20 family pairs; broken-link movement uses 20 trials. Each update uses its own model's context-specific baseline. Zero updates fail directional scoring.

Brackets are descriptive 95% intervals from 2,000 whole-family bootstrap draws (seed 20260921), preserving contexts and update arms. No paired between-model significance test is claimed.

| Model | Correct sign (%) [95% interval] | Paired reversal (%) [95% interval] | Broken mean absolute update (pp) [95% interval] |
|---|---:|---:|---:|
| Qwen2.5-72B | 62.5 [52.5, 72.5]; 25/40 | 25.0 [5.0, 45.0]; 5/20 | 12.8 [8.8, 16.8]; 20/20 |
| Llama-3.1-70B | 65.0 [52.5, 77.5]; 26/40 | 35.0 [15.0, 55.0]; 7/20 | 9.8 [3.5, 17.0]; 20/20 |
| Qwen3-32B | 62.5 [50.0, 75.0]; 25/40 | 30.0 [10.0, 50.0]; 6/20 | 13.2 [8.0, 18.8]; 20/20 |
| GPT-5.6 | 92.5 [85.0, 100.0]; 37/40 | 85.0 [70.0, 100.0]; 17/20 | 0.5 [0.0, 1.5]; 20/20 |

| Model | Broken stability criterion | No-news mean / mean absolute (pp) | Repeated-news mean / mean absolute (pp) |
|---|---|---:|---:|
| Qwen2.5-72B | criterion_not_met | 0.0000 / 0.0000 | 0.1375 / 0.2625 |
| Llama-3.1-70B | criterion_not_met | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Qwen3-32B | criterion_not_met | 0.0000 / 0.0000 | 0.5000 / 0.5000 |
| GPT-5.6 | criterion_met | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

Control drift pools all 80 context units per model. The broken-link criterion requires complete observations, a signed-mean 90% interval strictly within ±2 pp, and the upper 95% mean-absolute bound below 2 pp. Masked contexts have no prespecified direction or null target; context-specific controls and drift-adjusted results remain in comparison.json.

Provenance: local_repeat0_audit.json records exact equality of the original repeat-0 plan rows, source response line numbers and hashes, checks against each model's own baseline, and unchanged original full-plan input hashes. The frontier summary's recorded file hashes are verified and its entire analysis is recomputed before this report is generated. The original three-repeat local aggregate is preserved.

Regenerate offline from the repository root: `python exp1_prospective/context_reversal/summarize_frontier.py`.
