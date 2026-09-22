# Direction-only companion: completed local models

All **720/720** planned outputs are valid: three independent messages for each of 20 families × four contexts, using repeat 0. No numeric prior was displayed or forecast elicited. The original probability protocol and its primary outcomes remain unchanged.

| Model | Direction signs / 40 | Original numeric signs / 40 | Direction paired / 20 | Original numeric paired / 20 |
|---|---:|---:|---:|---:|
| Qwen2.5-72B | 25 | 25 | 6 | 5 |
| Llama3.1-70B | 31 | 26 | 11 | 7 |
| Qwen3-32B (nonthinking) | 28 | 25 | 9 | 6 |

| Model | Broken: unchanged / 20 | Controls: unchanged / 160 | Unclear / 240 |
|---|---:|---:|---:|
| Qwen2.5-72B | 6 | 150 | 0 |
| Llama3.1-70B | 7 | 155 | 1 |
| Qwen3-32B (nonthinking) | 2 | 150 | 2 |

Signed scoring requires increase in positive contexts and decrease in negative contexts. Paired scoring requires both within the same family. Broken new news and both control messages require unchanged. Masked new news is unscored. All planned observations remain in scored denominators; unclear, invalid, and missing outputs fail.

## Comparison with original numeric updates

| Model | Both correct | Classification only: endpoint blocked | Classification only: unblocked | Numeric only | Neither correct |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-72B | 22 | 0 | 3 | 3 | 12 |
| Llama3.1-70B | 26 | 4 | 1 | 0 | 9 |
| Qwen3-32B (nonthinking) | 23 | 4 | 1 | 2 | 10 |

**Most classification-only successes for Llama and Qwen3 occur at direction-blocking baseline endpoints: 4 of 5 for each model.** A baseline of 1 prevents a positive update, and a baseline of 0 prevents a negative update. Llama has six such signed trials; Qwen3 has four; Qwen72 has none.

Among unblocked trials, Llama scores 27/34 in classification versus 26/34 numerically; Qwen3 scores 24/36 versus 25/36; Qwen72 scores 25/40 in both tasks. Qwen72 has three classification-only and three numeric-only successes. The positive net gains for Llama and Qwen3 are therefore concentrated in endpoint cases. These separate task elicitations and one repeat do not establish the cause of those differences.

All five classification-only successes for each of Llama and Qwen3 have numeric baselines at 0 or 1. Their one unblocked case each is a positive target starting at 0, where an upward update was possible. Classification still performs poorly on broken-link unchanged judgments and misses many paired reversals.

For completeness, exact-zero numeric movement occurred on broken trials in Qwen2.5-72B: 5/20, Llama3.1-70B: 11/20, Qwen3-32B (nonthinking): 8/20; and on controls in Qwen2.5-72B: 156/160, Llama3.1-70B: 160/160, Qwen3-32B (nonthinking): 157/160. These exact-zero counts are descriptive companion diagnostics; the original interval-based stability analysis remains separate.

## Audit and provenance

Independent audits matched every raw JSON label, prompt and chat hash, request seed, model/input provenance, and all scoring and cross-tab denominators. Chat serialization was reconstructed using each local tokenizer without model inference. Original numeric comparisons use only repeat 0 from the existing full three-repeat runs.

- [Qwen72 independent audit](direction_qwen2_5_72b_instruct/independent_audit.json)
- [Llama70 independent audit](direction_llama3_1_70b_instruct/independent_audit.json)
- [Qwen3 independent audit](direction_qwen3_32b/independent_audit.json)
- [Machine-readable summary with source summary SHA-256 hashes](direction_summary.json)
