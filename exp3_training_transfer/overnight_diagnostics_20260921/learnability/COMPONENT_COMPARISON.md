# Fixed supervised learnability comparison

Updated 2026-09-22T14:43:46.201302+00:00.

The primary diagnostic comparison uses matched RL seed42 versus both fixed one-pass SFT recipes and the same untrained base on the frozen paired component diagnostic. SFT training used 4,800 original prompts, seed42, rank32/alpha64, one pass and 300 optimizer updates; RL sampled eight completions per prompt and used about 2,400 optimizer updates. The recipes match prompt exposures, not compute or update count. The second SFT learning rate differs from RL. These are outcome-motivated exploratory diagnostics, not a population-level or isolated universal SFT-versus-RL effect. Both fixed SFT recipes are retained. Matched RL seeds43/44 are included as descriptive context.

The original SFT launch gate was closed. A disclosed amendment justified the pilot from component evidence of stronger-model conditional arithmetic and an absolute-level interface error. See PROTOCOL.md and its gate/amendment records. Selection-only was already perfect for base/RL; any SFT improvement here concerns full forecasting/composition, not newly acquired cue-name identification.

## Original task

| Domain | Labels | Model/recipe | Response MAE | Cue calibration | Calibration 95% episode CI | Structure accuracy | Parse |
|---|---|---|---:|---:|---|---:|---:|
| coin_city | semantic | Base 8B | 2.105 | -0.211 | [-0.467, 0.014] | 0.458 | 1.000 |
| coin_city | semantic | Matched RL seed42 | 1.077 | 0.005 | [-0.074, 0.078] | 0.500 | 1.000 |
| coin_city | semantic | Matched RL seed43 (descriptive) | 1.029 | 0.036 | [-0.053, 0.113] | 0.521 | 1.000 |
| coin_city | semantic | Matched RL seed44 (descriptive) | 1.053 | -0.041 | [-0.107, 0.020] | 0.500 | 1.000 |
| coin_city | semantic | SFT forecast_lr1e6 seed42 | 1.253 | 0.166 | [-0.235, 0.549] | 0.583 | 1.000 |
| coin_city | semantic | SFT forecast_lr2e5 seed42 | 0.658 | 1.413 | [1.205, 1.620] | 0.979 | 1.000 |
| coin_city | arbitrary | Base 8B | 2.117 | 0.344 | [0.162, 0.529] | 0.542 | 1.000 |
| coin_city | arbitrary | Matched RL seed42 | 1.222 | 0.012 | [-0.068, 0.102] | 0.500 | 1.000 |
| coin_city | arbitrary | Matched RL seed43 (descriptive) | 1.096 | 0.016 | [-0.047, 0.078] | 0.438 | 1.000 |
| coin_city | arbitrary | Matched RL seed44 (descriptive) | 1.073 | 0.023 | [-0.034, 0.081] | 0.521 | 1.000 |
| coin_city | arbitrary | SFT forecast_lr1e6 seed42 | 1.727 | 0.001 | [-0.424, 0.489] | 0.438 | 1.000 |
| coin_city | arbitrary | SFT forecast_lr2e5 seed42 | 1.274 | -0.211 | [-0.516, 0.107] | 0.396 | 1.000 |
| coin_harbor | semantic | Base 8B | 4.967 | 0.170 | [-0.157, 0.499] | 0.438 | 1.000 |
| coin_harbor | semantic | Matched RL seed42 | 3.445 | 0.180 | [0.050, 0.325] | 0.500 | 1.000 |
| coin_harbor | semantic | Matched RL seed43 (descriptive) | 3.213 | 0.176 | [0.005, 0.363] | 0.500 | 1.000 |
| coin_harbor | semantic | Matched RL seed44 (descriptive) | 3.297 | 0.053 | [-0.049, 0.169] | 0.500 | 1.000 |
| coin_harbor | semantic | SFT forecast_lr1e6 seed42 | 5.317 | 0.058 | [-0.447, 0.538] | 0.542 | 1.000 |
| coin_harbor | semantic | SFT forecast_lr2e5 seed42 | 12.192 | -0.932 | [-1.528, -0.375] | 0.458 | 1.000 |
| coin_harbor | arbitrary | Base 8B | 5.353 | -0.094 | [-0.301, 0.109] | 0.500 | 1.000 |
| coin_harbor | arbitrary | Matched RL seed42 | 3.619 | 0.026 | [-0.115, 0.190] | 0.521 | 1.000 |
| coin_harbor | arbitrary | Matched RL seed43 (descriptive) | 3.098 | 0.067 | [-0.022, 0.157] | 0.583 | 1.000 |
| coin_harbor | arbitrary | Matched RL seed44 (descriptive) | 3.536 | -0.068 | [-0.259, 0.095] | 0.479 | 1.000 |
| coin_harbor | arbitrary | SFT forecast_lr1e6 seed42 | 5.965 | 0.120 | [-0.132, 0.381] | 0.479 | 1.000 |
| coin_harbor | arbitrary | SFT forecast_lr2e5 seed42 | 10.655 | 0.004 | [-0.282, 0.255] | 0.500 | 1.000 |

## Every component interface

| Model/recipe | Domain | Labels | Interface | Accuracy | Response MAE | Cue calibration |
|---|---|---|---|---:|---:|---:|
| Base 8B | coin_city | arbitrary | oracle_both_patterns | 0.500 | 2.710 | 0.217 |
| Base 8B | coin_city | arbitrary | oracle_selected_pattern | 0.396 | 2.961 | -0.219 |
| Base 8B | coin_city | arbitrary | original | 0.542 | 2.117 | 0.344 |
| Base 8B | coin_city | arbitrary | selection_only | 1.000 | — | — |
| Base 8B | coin_city | semantic | oracle_both_patterns | 0.479 | 2.393 | -0.066 |
| Base 8B | coin_city | semantic | oracle_selected_pattern | 0.458 | 2.413 | -0.010 |
| Base 8B | coin_city | semantic | original | 0.458 | 2.105 | -0.211 |
| Base 8B | coin_city | semantic | selection_only | 1.000 | — | — |
| Base 8B | coin_harbor | arbitrary | oracle_both_patterns | 0.500 | 12.155 | 0.120 |
| Base 8B | coin_harbor | arbitrary | oracle_selected_pattern | 0.271 | 8.826 | -1.944 |
| Base 8B | coin_harbor | arbitrary | original | 0.500 | 5.353 | -0.094 |
| Base 8B | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| Base 8B | coin_harbor | semantic | oracle_both_patterns | 0.542 | 7.841 | 0.227 |
| Base 8B | coin_harbor | semantic | oracle_selected_pattern | 0.479 | 7.872 | 0.259 |
| Base 8B | coin_harbor | semantic | original | 0.438 | 4.967 | 0.170 |
| Base 8B | coin_harbor | semantic | selection_only | 1.000 | — | — |
| Matched RL seed42 | coin_city | arbitrary | oracle_both_patterns | 0.208 | 3.831 | 1.241 |
| Matched RL seed42 | coin_city | arbitrary | oracle_selected_pattern | 0.500 | 1.563 | 0.066 |
| Matched RL seed42 | coin_city | arbitrary | original | 0.500 | 1.222 | 0.012 |
| Matched RL seed42 | coin_city | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed42 | coin_city | semantic | oracle_both_patterns | 0.417 | 2.702 | 0.406 |
| Matched RL seed42 | coin_city | semantic | oracle_selected_pattern | 0.500 | 1.467 | -0.143 |
| Matched RL seed42 | coin_city | semantic | original | 0.500 | 1.077 | 0.005 |
| Matched RL seed42 | coin_city | semantic | selection_only | 1.000 | — | — |
| Matched RL seed42 | coin_harbor | arbitrary | oracle_both_patterns | 0.458 | 9.328 | 0.132 |
| Matched RL seed42 | coin_harbor | arbitrary | oracle_selected_pattern | 0.271 | 9.449 | -1.018 |
| Matched RL seed42 | coin_harbor | arbitrary | original | 0.521 | 3.619 | 0.026 |
| Matched RL seed42 | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed42 | coin_harbor | semantic | oracle_both_patterns | 0.479 | 8.663 | 0.190 |
| Matched RL seed42 | coin_harbor | semantic | oracle_selected_pattern | 0.271 | 8.411 | -1.181 |
| Matched RL seed42 | coin_harbor | semantic | original | 0.500 | 3.445 | 0.180 |
| Matched RL seed42 | coin_harbor | semantic | selection_only | 1.000 | — | — |
| Matched RL seed43 (descriptive) | coin_city | arbitrary | oracle_both_patterns | 0.250 | 4.497 | -0.432 |
| Matched RL seed43 (descriptive) | coin_city | arbitrary | oracle_selected_pattern | 0.500 | 1.573 | -0.012 |
| Matched RL seed43 (descriptive) | coin_city | arbitrary | original | 0.438 | 1.096 | 0.016 |
| Matched RL seed43 (descriptive) | coin_city | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed43 (descriptive) | coin_city | semantic | oracle_both_patterns | 0.458 | 2.404 | 0.115 |
| Matched RL seed43 (descriptive) | coin_city | semantic | oracle_selected_pattern | 0.521 | 1.389 | 0.107 |
| Matched RL seed43 (descriptive) | coin_city | semantic | original | 0.521 | 1.029 | 0.036 |
| Matched RL seed43 (descriptive) | coin_city | semantic | selection_only | 1.000 | — | — |
| Matched RL seed43 (descriptive) | coin_harbor | arbitrary | oracle_both_patterns | 0.438 | 16.463 | 0.357 |
| Matched RL seed43 (descriptive) | coin_harbor | arbitrary | oracle_selected_pattern | 0.333 | 9.073 | -0.939 |
| Matched RL seed43 (descriptive) | coin_harbor | arbitrary | original | 0.583 | 3.098 | 0.067 |
| Matched RL seed43 (descriptive) | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed43 (descriptive) | coin_harbor | semantic | oracle_both_patterns | 0.479 | 13.316 | -0.735 |
| Matched RL seed43 (descriptive) | coin_harbor | semantic | oracle_selected_pattern | 0.292 | 10.289 | -1.086 |
| Matched RL seed43 (descriptive) | coin_harbor | semantic | original | 0.500 | 3.213 | 0.176 |
| Matched RL seed43 (descriptive) | coin_harbor | semantic | selection_only | 1.000 | — | — |
| Matched RL seed44 (descriptive) | coin_city | arbitrary | oracle_both_patterns | 0.417 | 3.052 | 0.258 |
| Matched RL seed44 (descriptive) | coin_city | arbitrary | oracle_selected_pattern | 0.521 | 1.565 | 0.063 |
| Matched RL seed44 (descriptive) | coin_city | arbitrary | original | 0.521 | 1.073 | 0.023 |
| Matched RL seed44 (descriptive) | coin_city | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed44 (descriptive) | coin_city | semantic | oracle_both_patterns | 0.500 | 1.532 | 0.122 |
| Matched RL seed44 (descriptive) | coin_city | semantic | oracle_selected_pattern | 0.500 | 1.464 | -0.066 |
| Matched RL seed44 (descriptive) | coin_city | semantic | original | 0.500 | 1.053 | -0.041 |
| Matched RL seed44 (descriptive) | coin_city | semantic | selection_only | 1.000 | — | — |
| Matched RL seed44 (descriptive) | coin_harbor | arbitrary | oracle_both_patterns | 0.458 | 15.720 | -0.130 |
| Matched RL seed44 (descriptive) | coin_harbor | arbitrary | oracle_selected_pattern | 0.292 | 9.387 | -1.253 |
| Matched RL seed44 (descriptive) | coin_harbor | arbitrary | original | 0.479 | 3.536 | -0.068 |
| Matched RL seed44 (descriptive) | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| Matched RL seed44 (descriptive) | coin_harbor | semantic | oracle_both_patterns | 0.458 | 6.992 | -0.351 |
| Matched RL seed44 (descriptive) | coin_harbor | semantic | oracle_selected_pattern | 0.333 | 7.198 | -0.640 |
| Matched RL seed44 (descriptive) | coin_harbor | semantic | original | 0.500 | 3.297 | 0.053 |
| Matched RL seed44 (descriptive) | coin_harbor | semantic | selection_only | 1.000 | — | — |
| SFT forecast_lr1e6 seed42 | coin_city | arbitrary | oracle_both_patterns | 0.167 | 2.186 | 0.051 |
| SFT forecast_lr1e6 seed42 | coin_city | arbitrary | oracle_selected_pattern | 0.229 | 2.238 | 0.745 |
| SFT forecast_lr1e6 seed42 | coin_city | arbitrary | original | 0.438 | 1.727 | 0.001 |
| SFT forecast_lr1e6 seed42 | coin_city | arbitrary | selection_only | 1.000 | — | — |
| SFT forecast_lr1e6 seed42 | coin_city | semantic | oracle_both_patterns | 0.208 | 2.334 | 0.774 |
| SFT forecast_lr1e6 seed42 | coin_city | semantic | oracle_selected_pattern | 0.375 | 1.932 | 0.369 |
| SFT forecast_lr1e6 seed42 | coin_city | semantic | original | 0.583 | 1.253 | 0.166 |
| SFT forecast_lr1e6 seed42 | coin_city | semantic | selection_only | 1.000 | — | — |
| SFT forecast_lr1e6 seed42 | coin_harbor | arbitrary | oracle_both_patterns | 0.500 | 5.507 | 0.000 |
| SFT forecast_lr1e6 seed42 | coin_harbor | arbitrary | oracle_selected_pattern | 0.500 | 5.579 | -0.068 |
| SFT forecast_lr1e6 seed42 | coin_harbor | arbitrary | original | 0.479 | 5.965 | 0.120 |
| SFT forecast_lr1e6 seed42 | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| SFT forecast_lr1e6 seed42 | coin_harbor | semantic | oracle_both_patterns | 0.458 | 7.187 | 0.157 |
| SFT forecast_lr1e6 seed42 | coin_harbor | semantic | oracle_selected_pattern | 0.500 | 9.400 | 0.324 |
| SFT forecast_lr1e6 seed42 | coin_harbor | semantic | original | 0.542 | 5.317 | 0.058 |
| SFT forecast_lr1e6 seed42 | coin_harbor | semantic | selection_only | 1.000 | — | — |
| SFT forecast_lr2e5 seed42 | coin_city | arbitrary | oracle_both_patterns | 0.500 | 2.253 | -0.015 |
| SFT forecast_lr2e5 seed42 | coin_city | arbitrary | oracle_selected_pattern | 0.542 | 1.981 | 0.273 |
| SFT forecast_lr2e5 seed42 | coin_city | arbitrary | original | 0.396 | 1.274 | -0.211 |
| SFT forecast_lr2e5 seed42 | coin_city | arbitrary | selection_only | 1.000 | — | — |
| SFT forecast_lr2e5 seed42 | coin_city | semantic | oracle_both_patterns | 0.604 | 1.669 | 0.474 |
| SFT forecast_lr2e5 seed42 | coin_city | semantic | oracle_selected_pattern | 0.875 | 1.117 | 1.320 |
| SFT forecast_lr2e5 seed42 | coin_city | semantic | original | 0.979 | 0.658 | 1.413 |
| SFT forecast_lr2e5 seed42 | coin_city | semantic | selection_only | 1.000 | — | — |
| SFT forecast_lr2e5 seed42 | coin_harbor | arbitrary | oracle_both_patterns | 0.521 | 12.912 | 0.433 |
| SFT forecast_lr2e5 seed42 | coin_harbor | arbitrary | oracle_selected_pattern | 0.521 | 10.955 | 0.320 |
| SFT forecast_lr2e5 seed42 | coin_harbor | arbitrary | original | 0.500 | 10.655 | 0.004 |
| SFT forecast_lr2e5 seed42 | coin_harbor | arbitrary | selection_only | 1.000 | — | — |
| SFT forecast_lr2e5 seed42 | coin_harbor | semantic | oracle_both_patterns | 0.542 | 13.702 | -0.097 |
| SFT forecast_lr2e5 seed42 | coin_harbor | semantic | oracle_selected_pattern | 0.521 | 11.527 | 0.015 |
| SFT forecast_lr2e5 seed42 | coin_harbor | semantic | original | 0.458 | 12.192 | -0.932 |
| SFT forecast_lr2e5 seed42 | coin_harbor | semantic | selection_only | 1.000 | — | — |

Calibration CIs resample paired episodes within a cell. They do not account for training-seed variation and must not be used for a seed-generalized method claim. The original scoring clips forecast values to the domain range before forming response vectors; baseline/raw sensitivities should be examined if an SFT output omits the resting level.
