# Structure component diagnostic results

Updated 2026-09-22T03:46:16.165676+00:00. 8/8 frozen model endpoints complete.

Fresh fixed evaluation: 96 paired latent episodes, 24 per domain × label cell; zero target observations; 768 prompts per model. Greedy JSON-constrained inference. Selection accuracy tests cue-to-reference identification. Forecast response MAE measures numerical forecast quality. Cue-change calibration is 0 for cue-invariant forecasts and 1 for the simulator-predicted paired change. These are diagnostic results, not the registered multi-draw endpoint or a seed-level significance claim.

## Aggregate overview

| Model | Interface | Parse | Accuracy | Response MAE | Cue-change calibration |
|---|---|---:|---:|---:|---:|
| qwen3_32b_base | selection_only | 1.000 | 1.000 | — | — |
| qwen3_32b_base | original | 1.000 | 0.365 | 5.334 | -0.027 |
| qwen3_32b_base | oracle_both_patterns | 1.000 | 0.380 | 3.559 | 0.098 |
| qwen3_32b_base | oracle_selected_pattern | 1.000 | 0.490 | 3.239 | 0.326 |
| qwen3_8b_base | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_base | original | 1.000 | 0.484 | 3.636 | 0.052 |
| qwen3_8b_base | oracle_both_patterns | 1.000 | 0.505 | 6.275 | 0.124 |
| qwen3_8b_base | oracle_selected_pattern | 1.000 | 0.401 | 5.518 | -0.478 |
| qwen3_8b_causal_s42 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_causal_s42 | original | 1.000 | 0.505 | 2.341 | 0.056 |
| qwen3_8b_causal_s42 | oracle_both_patterns | 1.000 | 0.391 | 6.131 | 0.492 |
| qwen3_8b_causal_s42 | oracle_selected_pattern | 1.000 | 0.385 | 5.223 | -0.569 |
| qwen3_8b_causal_s43 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_causal_s43 | original | 1.000 | 0.510 | 2.109 | 0.074 |
| qwen3_8b_causal_s43 | oracle_both_patterns | 1.000 | 0.406 | 9.170 | -0.174 |
| qwen3_8b_causal_s43 | oracle_selected_pattern | 1.000 | 0.411 | 5.581 | -0.482 |
| qwen3_8b_causal_s44 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_causal_s44 | original | 1.000 | 0.500 | 2.240 | -0.008 |
| qwen3_8b_causal_s44 | oracle_both_patterns | 1.000 | 0.458 | 6.824 | -0.025 |
| qwen3_8b_causal_s44 | oracle_selected_pattern | 1.000 | 0.411 | 4.904 | -0.474 |
| qwen3_8b_population_prior_s42 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_population_prior_s42 | original | 1.000 | 0.505 | 2.005 | 0.033 |
| qwen3_8b_population_prior_s42 | oracle_both_patterns | 1.000 | 0.370 | 8.276 | 0.803 |
| qwen3_8b_population_prior_s42 | oracle_selected_pattern | 1.000 | 0.432 | 4.144 | -0.006 |
| qwen3_8b_population_prior_s43 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_population_prior_s43 | original | 1.000 | 0.521 | 2.024 | 0.042 |
| qwen3_8b_population_prior_s43 | oracle_both_patterns | 1.000 | 0.333 | 5.321 | 0.109 |
| qwen3_8b_population_prior_s43 | oracle_selected_pattern | 1.000 | 0.432 | 4.927 | -0.148 |
| qwen3_8b_population_prior_s44 | selection_only | 1.000 | 1.000 | — | — |
| qwen3_8b_population_prior_s44 | original | 1.000 | 0.495 | 1.965 | 0.030 |
| qwen3_8b_population_prior_s44 | oracle_both_patterns | 1.000 | 0.385 | 6.444 | 0.654 |
| qwen3_8b_population_prior_s44 | oracle_selected_pattern | 1.000 | 0.427 | 4.420 | -0.018 |

Accuracy is reference identification for selection-only; nearest-template structure accuracy for forecasts. Overview averages the four equally sized domain × label cells. See below for transfer boundaries.

## qwen3_32b_base

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.354 | 1.993 | 0.148 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.354 | 1.604 | 0.263 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.417 | 2.671 | 0.011 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.125 | 1.710 | 0.018 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.500 | 1.423 | 0.688 | 24 |
| coin_city | semantic | original | 1.000 | 0.083 | 4.110 | -0.961 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.542 | 5.242 | 0.149 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.542 | 5.200 | 0.188 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.500 | 7.012 | 0.012 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.500 | 5.288 | 0.079 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.562 | 4.730 | 0.166 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.458 | 7.543 | 0.830 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_base

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.500 | 2.710 | 0.217 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.396 | 2.961 | -0.219 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.542 | 2.117 | 0.344 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.479 | 2.393 | -0.066 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.458 | 2.413 | -0.010 | 24 |
| coin_city | semantic | original | 1.000 | 0.458 | 2.105 | -0.211 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.500 | 12.155 | 0.120 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.271 | 8.826 | -1.944 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.500 | 5.353 | -0.094 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.542 | 7.841 | 0.227 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.479 | 7.872 | 0.259 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.438 | 4.967 | 0.170 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_causal_s42

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.208 | 3.831 | 1.241 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.500 | 1.563 | 0.066 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.500 | 1.222 | 0.012 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.417 | 2.702 | 0.406 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.500 | 1.467 | -0.143 | 24 |
| coin_city | semantic | original | 1.000 | 0.500 | 1.077 | 0.005 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.458 | 9.328 | 0.132 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.271 | 9.449 | -1.018 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.521 | 3.619 | 0.026 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.479 | 8.663 | 0.190 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.271 | 8.411 | -1.181 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 3.445 | 0.180 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_causal_s43

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.250 | 4.497 | -0.432 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.500 | 1.573 | -0.012 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.438 | 1.096 | 0.016 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.458 | 2.404 | 0.115 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.521 | 1.389 | 0.107 | 24 |
| coin_city | semantic | original | 1.000 | 0.521 | 1.029 | 0.036 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.438 | 16.463 | 0.357 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.333 | 9.073 | -0.939 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.583 | 3.098 | 0.067 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.479 | 13.316 | -0.735 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.292 | 10.289 | -1.086 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 3.213 | 0.176 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_causal_s44

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.417 | 3.052 | 0.258 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.521 | 1.565 | 0.063 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.521 | 1.073 | 0.023 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.500 | 1.532 | 0.122 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.500 | 1.464 | -0.066 | 24 |
| coin_city | semantic | original | 1.000 | 0.500 | 1.053 | -0.041 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.458 | 15.720 | -0.130 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.292 | 9.387 | -1.253 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.479 | 3.536 | -0.068 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.458 | 6.992 | -0.351 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.333 | 7.198 | -0.640 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 3.297 | 0.053 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_population_prior_s42

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.208 | 4.488 | 0.043 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.521 | 1.343 | 0.040 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.542 | 1.094 | 0.009 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.271 | 3.707 | 1.678 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.500 | 1.161 | 0.162 | 24 |
| coin_city | semantic | original | 1.000 | 0.479 | 0.992 | -0.014 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.562 | 16.475 | 1.610 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.333 | 7.669 | -0.558 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.500 | 3.088 | 0.014 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.438 | 8.433 | -0.121 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.375 | 6.401 | 0.330 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 2.846 | 0.122 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_population_prior_s43

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.125 | 3.430 | -0.009 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.542 | 1.368 | 0.096 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.500 | 1.091 | 0.024 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.250 | 3.601 | 0.451 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.521 | 1.252 | 0.101 | 24 |
| coin_city | semantic | original | 1.000 | 0.521 | 1.015 | -0.006 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.479 | 7.614 | 0.093 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.312 | 8.694 | -0.699 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.562 | 3.011 | 0.063 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.479 | 6.640 | -0.101 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.354 | 8.393 | -0.091 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 2.980 | 0.089 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## qwen3_8b_population_prior_s44

| Domain | Labels | Interface | Parse | Accuracy | Response MAE | Calibration | Valid pairs |
|---|---|---|---:|---:|---:|---:|---:|
| coin_city | arbitrary | oracle_both_patterns | 1.000 | 0.208 | 4.550 | -0.187 | 24 |
| coin_city | arbitrary | oracle_selected_pattern | 1.000 | 0.500 | 1.315 | 0.140 | 24 |
| coin_city | arbitrary | original | 1.000 | 0.521 | 1.068 | 0.047 | 24 |
| coin_city | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_city | semantic | oracle_both_patterns | 1.000 | 0.292 | 3.663 | 1.779 | 24 |
| coin_city | semantic | oracle_selected_pattern | 1.000 | 0.521 | 1.258 | 0.070 | 24 |
| coin_city | semantic | original | 1.000 | 0.438 | 0.964 | -0.002 | 24 |
| coin_city | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | arbitrary | oracle_both_patterns | 1.000 | 0.583 | 8.823 | 0.236 | 24 |
| coin_harbor | arbitrary | oracle_selected_pattern | 1.000 | 0.354 | 8.309 | -0.699 | 24 |
| coin_harbor | arbitrary | original | 1.000 | 0.521 | 2.945 | 0.029 | 24 |
| coin_harbor | arbitrary | selection_only | 1.000 | 1.000 | — | — | 24 |
| coin_harbor | semantic | oracle_both_patterns | 1.000 | 0.458 | 8.741 | 0.788 | 24 |
| coin_harbor | semantic | oracle_selected_pattern | 1.000 | 0.333 | 6.798 | 0.415 | 24 |
| coin_harbor | semantic | original | 1.000 | 0.500 | 2.883 | 0.045 | 24 |
| coin_harbor | semantic | selection_only | 1.000 | 1.000 | — | — | 24 |

## Coverage and limitations

Completed: qwen3_32b_base, qwen3_8b_base, qwen3_8b_causal_s42, qwen3_8b_causal_s43, qwen3_8b_causal_s44, qwen3_8b_population_prior_s42, qwen3_8b_population_prior_s43, qwen3_8b_population_prior_s44.
Remaining: none.

The oracle gives exact latent per-unit effects, while original trajectories are noisy. The diagnostic therefore tests whether numerical assistance removes a bottleneck; it does not claim those coefficients can be estimated exactly from fourteen rows. There are no target observations, so target-evidence integration remains outside this diagnostic. Cell CIs and invalid-generation penalties are recorded in the JSON summaries. See PROTOCOL.md and amendment_01_oracle_clarity.json for the frozen design and preregeneration wording correction.
