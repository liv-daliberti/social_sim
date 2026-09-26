# Secondary saturation diagnostic

Computed from saved responses only; no new model inference. This is a post hoc,
model-dependent feasibility diagnostic, not a replacement for unconditional
primary scores and not a fairer confirmatory model ranking. A positive update is
blocked at baseline 1, and a negative update at baseline 0. Near-endpoint priors
are a separate issue not resolved by this exact-endpoint calculation.

| Model | Blocked directions, all local repeats | Correct signs among feasible | Both signs correct among feasible pairs |
|---|---:|---:|---:|
| Qwen2.5-72B | 0/120 | 75/120 (62.5%) | 18/60 (30.0%) |
| Llama-3.1-70B | 19/120 | 78/101 (77.2%) | 22/42 (52.4%) |
| Qwen3-32B | 12/120 | 77/108 (71.3%) | 19/48 (39.6%) |
| GPT-5.6, one repeat | 0/40 | 37/40 (92.5%) | 17/20 (85.0%) |

On matched repeat zero, feasible paired reversal is 5/20 for Qwen2.5, 7/14
for Llama, 6/16 for Qwen3, and 17/20 for GPT. The selected sets differ by model;
these percentages diagnose saturation rather than estimate a controlled model
effect. Saturation accounts for part of the local failure but cannot account for
all of it. All original primary scores, labels, materials, and manuscript claims
remain unchanged.

Next development priorities are independent review of all families, new-version
repairs of ambiguous routes, a local direction-only companion task, and a local
Qwen3 reasoning-enabled arm. The last requires a separate runner configuration
that permits reasoning before extracting the final probability; enabling a flag
under the original probability-only regex is insufficient.
