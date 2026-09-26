# Coin City symbol-context comparison — interim local results

Status: Qwen2.5-32B and Qwen2.5-72B pending. Exploratory only; not included in the paper.

Primary stratum: k=0 (no City C demonstrations), 250 paired episodes per model. Bootstrap draws: 2,000.

| Model | No-context MAE | Semantic MAE | Symbol MAE | Symbol − no-context ΔMAE [95% CI] | No-context ρ | Symbol ρ | Symbol − no-context Δρ [95% CI] | No-context acc. | Symbol acc. | Δacc. [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 2.896 | 2.365 | 2.892 | -0.005 [-0.088, 0.074] | -0.011 | 0.085 | 0.095 [-0.060, 0.230] | 0.496 | 0.512 | 0.016 [-0.008, 0.044] |
| Qwen2.5-7B | 2.911 | 2.669 | 3.248 | 0.338 [0.027, 0.682] | -0.034 | 0.230 | 0.264 [0.112, 0.406] | 0.500 | 0.572 | 0.072 [0.020, 0.128] |
| Qwen3-8B | 2.985 | 2.859 | 2.973 | -0.011 [-0.239, 0.207] | -0.014 | 0.009 | 0.023 [-0.128, 0.171] | 0.492 | 0.516 | 0.024 [-0.012, 0.060] |
| Qwen3-14B | 2.632 | 2.241 | 2.812 | 0.180 [-0.188, 0.548] | -0.057 | 0.342 | 0.398 [0.253, 0.541] | 0.496 | 0.660 | 0.164 [0.088, 0.240] |
| DeepSeek V4 Pro | 2.622 | 2.588 | 2.729 | 0.107 [-0.222, 0.452] | 0.018 | 0.454 | 0.436 [0.323, 0.558] | 0.520 | 0.692 | 0.172 [0.104, 0.240] |
| Claude Opus 5 | 2.619 | 1.764 | 2.017 | -0.603 [-0.869, -0.336] | -0.007 | 0.583 | 0.591 [0.485, 0.697] | 0.496 | 0.780 | 0.284 [0.220, 0.348] |
| GPT-5.6 Sol | 2.518 | 1.596 | 1.744 | -0.774 [-1.011, -0.533] | -0.041 | 0.655 | 0.696 [0.597, 0.807] | 0.464 | 0.836 | 0.372 [0.312, 0.432] |

Interpretation pending the large-Qwen arms: arbitrary-label discrimination is absent at Qwen3-4B/8B, emerges strongly at Qwen3-14B, is modest at Qwen2.5-7B but worsens absolute MAE, and is strongest with meaningful MAE gains for Claude Opus 5 and GPT-5.6 Sol.

Original model responses are preserved. Arithmetic-only reparsing uses no additional model calls; omitted prediction fields remain unresolved.
