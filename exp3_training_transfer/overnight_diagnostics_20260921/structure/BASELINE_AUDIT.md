# Post hoc baseline and raw-response sensitivity

Post hoc sensitivity motivated by stronger-model forecasts omitting the resting baseline. Frozen primary metrics unchanged.

This audit subtracts each horizon's predicted zero-shock output before calculating response error, without clipping predictions first. It separates an omitted resting level from incorrect response patterns. Exact relative accuracy means all eight response coordinates are within 0.01. Offset-corrected MAE adds one common shift estimated from the two zero-shock predictions; it is an interpretive sensitivity, not a valid submitted forecast or replacement primary outcome.

| Model | Domain | Labels | Interface | Raw response MAE | Relative exact | Baseline omitted | Offset-corrected MAE | Raw cue calibration |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3_32b_base | coin_city | arbitrary | oracle_both_patterns | 1.225 | 0.354 | 1.000 | 0.980 | 0.295 |
| qwen3_32b_base | coin_city | arbitrary | oracle_selected_pattern | 0.525 | 0.417 | 0.917 | 0.420 | 0.583 |
| qwen3_32b_base | coin_city | arbitrary | original | 2.721 | 0.000 | 0.062 | 2.177 | 0.045 |
| qwen3_32b_base | coin_city | semantic | oracle_both_patterns | 0.883 | 0.396 | 1.000 | 0.707 | 0.037 |
| qwen3_32b_base | coin_city | semantic | oracle_selected_pattern | 0.384 | 0.521 | 0.958 | 0.307 | 1.262 |
| qwen3_32b_base | coin_city | semantic | original | 4.794 | 0.000 | 0.604 | 4.831 | 0.090 |
| qwen3_32b_base | coin_harbor | arbitrary | oracle_both_patterns | 1.500 | 0.375 | 0.938 | 1.200 | 0.370 |
| qwen3_32b_base | coin_harbor | arbitrary | oracle_selected_pattern | 2.494 | 0.354 | 0.854 | 1.995 | 1.865 |
| qwen3_32b_base | coin_harbor | arbitrary | original | 7.652 | 0.000 | 0.000 | 6.122 | -0.035 |
| qwen3_32b_base | coin_harbor | semantic | oracle_both_patterns | 2.910 | 0.479 | 1.000 | 2.328 | 1.253 |
| qwen3_32b_base | coin_harbor | semantic | oracle_selected_pattern | 3.187 | 0.188 | 0.854 | 2.550 | 1.645 |
| qwen3_32b_base | coin_harbor | semantic | original | 10.754 | 0.000 | 0.167 | 10.453 | 1.385 |
| qwen3_8b_base | coin_city | arbitrary | oracle_both_patterns | 2.710 | 0.000 | 0.000 | 2.168 | 0.217 |
| qwen3_8b_base | coin_city | arbitrary | oracle_selected_pattern | 3.489 | 0.000 | 0.354 | 2.791 | -0.748 |
| qwen3_8b_base | coin_city | arbitrary | original | 2.117 | 0.000 | 0.000 | 1.694 | 0.344 |
| qwen3_8b_base | coin_city | semantic | oracle_both_patterns | 2.393 | 0.000 | 0.000 | 1.914 | -0.066 |
| qwen3_8b_base | coin_city | semantic | oracle_selected_pattern | 2.431 | 0.000 | 0.021 | 1.945 | 0.084 |
| qwen3_8b_base | coin_city | semantic | original | 2.105 | 0.000 | 0.000 | 1.691 | -0.211 |
| qwen3_8b_base | coin_harbor | arbitrary | oracle_both_patterns | 23.248 | 0.000 | 0.938 | 18.598 | 0.171 |
| qwen3_8b_base | coin_harbor | arbitrary | oracle_selected_pattern | 10.855 | 0.000 | 0.708 | 8.684 | -0.123 |
| qwen3_8b_base | coin_harbor | arbitrary | original | 5.353 | 0.000 | 0.000 | 4.326 | -0.094 |
| qwen3_8b_base | coin_harbor | semantic | oracle_both_patterns | 7.966 | 0.000 | 0.104 | 6.373 | -0.008 |
| qwen3_8b_base | coin_harbor | semantic | oracle_selected_pattern | 8.243 | 0.000 | 0.396 | 6.595 | 0.358 |
| qwen3_8b_base | coin_harbor | semantic | original | 4.967 | 0.000 | 0.000 | 3.995 | 0.170 |
| qwen3_8b_causal_s42 | coin_city | arbitrary | oracle_both_patterns | 5.001 | 0.042 | 0.771 | 4.001 | 2.278 |
| qwen3_8b_causal_s42 | coin_city | arbitrary | oracle_selected_pattern | 1.563 | 0.000 | 0.000 | 1.257 | 0.066 |
| qwen3_8b_causal_s42 | coin_city | arbitrary | original | 1.222 | 0.000 | 0.000 | 1.010 | 0.012 |
| qwen3_8b_causal_s42 | coin_city | semantic | oracle_both_patterns | 2.916 | 0.000 | 0.188 | 2.497 | 0.832 |
| qwen3_8b_causal_s42 | coin_city | semantic | oracle_selected_pattern | 1.467 | 0.000 | 0.000 | 1.174 | -0.143 |
| qwen3_8b_causal_s42 | coin_city | semantic | original | 1.077 | 0.000 | 0.000 | 0.892 | 0.005 |
| qwen3_8b_causal_s42 | coin_harbor | arbitrary | oracle_both_patterns | 12.151 | 0.021 | 0.771 | 9.995 | 0.722 |
| qwen3_8b_causal_s42 | coin_harbor | arbitrary | oracle_selected_pattern | 10.004 | 0.000 | 0.333 | 8.003 | -0.134 |
| qwen3_8b_causal_s42 | coin_harbor | arbitrary | original | 3.619 | 0.000 | 0.000 | 2.935 | 0.026 |
| qwen3_8b_causal_s42 | coin_harbor | semantic | oracle_both_patterns | 14.539 | 0.000 | 0.896 | 14.024 | -0.586 |
| qwen3_8b_causal_s42 | coin_harbor | semantic | oracle_selected_pattern | 10.161 | 0.000 | 0.271 | 10.915 | 1.401 |
| qwen3_8b_causal_s42 | coin_harbor | semantic | original | 3.445 | 0.000 | 0.000 | 2.781 | 0.180 |
| qwen3_8b_causal_s43 | coin_city | arbitrary | oracle_both_patterns | 6.062 | 0.000 | 0.625 | 4.879 | -0.317 |
| qwen3_8b_causal_s43 | coin_city | arbitrary | oracle_selected_pattern | 1.573 | 0.000 | 0.000 | 1.266 | -0.012 |
| qwen3_8b_causal_s43 | coin_city | arbitrary | original | 1.096 | 0.000 | 0.000 | 0.955 | 0.016 |
| qwen3_8b_causal_s43 | coin_city | semantic | oracle_both_patterns | 2.285 | 0.000 | 0.062 | 2.452 | 0.170 |
| qwen3_8b_causal_s43 | coin_city | semantic | oracle_selected_pattern | 1.389 | 0.000 | 0.000 | 1.117 | 0.107 |
| qwen3_8b_causal_s43 | coin_city | semantic | original | 1.029 | 0.000 | 0.000 | 0.868 | 0.036 |
| qwen3_8b_causal_s43 | coin_harbor | arbitrary | oracle_both_patterns | 25.720 | 0.021 | 0.771 | 20.850 | 1.062 |
| qwen3_8b_causal_s43 | coin_harbor | arbitrary | oracle_selected_pattern | 9.510 | 0.021 | 0.542 | 7.608 | -0.156 |
| qwen3_8b_causal_s43 | coin_harbor | arbitrary | original | 3.098 | 0.000 | 0.000 | 2.556 | 0.067 |
| qwen3_8b_causal_s43 | coin_harbor | semantic | oracle_both_patterns | 23.575 | 0.000 | 0.771 | 24.873 | -1.459 |
| qwen3_8b_causal_s43 | coin_harbor | semantic | oracle_selected_pattern | 14.454 | 0.000 | 0.208 | 17.471 | 0.598 |
| qwen3_8b_causal_s43 | coin_harbor | semantic | original | 3.213 | 0.000 | 0.000 | 2.595 | 0.176 |
| qwen3_8b_causal_s44 | coin_city | arbitrary | oracle_both_patterns | 3.752 | 0.000 | 0.312 | 3.002 | 0.471 |
| qwen3_8b_causal_s44 | coin_city | arbitrary | oracle_selected_pattern | 1.565 | 0.000 | 0.000 | 1.259 | 0.063 |
| qwen3_8b_causal_s44 | coin_city | arbitrary | original | 1.073 | 0.000 | 0.000 | 0.907 | 0.023 |
| qwen3_8b_causal_s44 | coin_city | semantic | oracle_both_patterns | 1.532 | 0.000 | 0.000 | 1.226 | 0.122 |
| qwen3_8b_causal_s44 | coin_city | semantic | oracle_selected_pattern | 1.464 | 0.000 | 0.000 | 1.172 | -0.066 |
| qwen3_8b_causal_s44 | coin_city | semantic | original | 1.053 | 0.000 | 0.000 | 0.875 | -0.041 |
| qwen3_8b_causal_s44 | coin_harbor | arbitrary | oracle_both_patterns | 17.576 | 0.021 | 0.854 | 14.061 | -0.342 |
| qwen3_8b_causal_s44 | coin_harbor | arbitrary | oracle_selected_pattern | 8.560 | 0.000 | 0.354 | 6.850 | -0.016 |
| qwen3_8b_causal_s44 | coin_harbor | arbitrary | original | 3.536 | 0.000 | 0.000 | 2.857 | -0.068 |
| qwen3_8b_causal_s44 | coin_harbor | semantic | oracle_both_patterns | 10.017 | 0.000 | 0.938 | 10.397 | -2.183 |
| qwen3_8b_causal_s44 | coin_harbor | semantic | oracle_selected_pattern | 6.778 | 0.000 | 0.167 | 5.422 | 0.157 |
| qwen3_8b_causal_s44 | coin_harbor | semantic | original | 3.297 | 0.000 | 0.000 | 2.669 | 0.053 |
| qwen3_8b_population_prior_s42 | coin_city | arbitrary | oracle_both_patterns | 6.127 | 0.000 | 0.708 | 5.860 | 0.117 |
| qwen3_8b_population_prior_s42 | coin_city | arbitrary | oracle_selected_pattern | 1.343 | 0.000 | 0.000 | 1.075 | 0.040 |
| qwen3_8b_population_prior_s42 | coin_city | arbitrary | original | 1.094 | 0.000 | 0.000 | 0.912 | 0.009 |
| qwen3_8b_population_prior_s42 | coin_city | semantic | oracle_both_patterns | 5.172 | 0.000 | 0.333 | 4.138 | 3.660 |
| qwen3_8b_population_prior_s42 | coin_city | semantic | oracle_selected_pattern | 1.161 | 0.000 | 0.000 | 0.929 | 0.162 |
| qwen3_8b_population_prior_s42 | coin_city | semantic | original | 0.992 | 0.000 | 0.000 | 0.829 | -0.014 |
| qwen3_8b_population_prior_s42 | coin_harbor | arbitrary | oracle_both_patterns | 23.580 | 0.000 | 0.521 | 18.864 | 1.122 |
| qwen3_8b_population_prior_s42 | coin_harbor | arbitrary | oracle_selected_pattern | 6.751 | 0.000 | 0.479 | 5.496 | 0.646 |
| qwen3_8b_population_prior_s42 | coin_harbor | arbitrary | original | 3.088 | 0.000 | 0.000 | 2.502 | 0.014 |
| qwen3_8b_population_prior_s42 | coin_harbor | semantic | oracle_both_patterns | 11.546 | 0.000 | 0.771 | 11.966 | 0.090 |
| qwen3_8b_population_prior_s42 | coin_harbor | semantic | oracle_selected_pattern | 6.093 | 0.000 | 0.208 | 6.098 | 1.053 |
| qwen3_8b_population_prior_s42 | coin_harbor | semantic | original | 2.846 | 0.000 | 0.000 | 2.319 | 0.122 |
| qwen3_8b_population_prior_s43 | coin_city | arbitrary | oracle_both_patterns | 3.772 | 0.021 | 0.875 | 3.018 | 0.157 |
| qwen3_8b_population_prior_s43 | coin_city | arbitrary | oracle_selected_pattern | 1.368 | 0.000 | 0.000 | 1.094 | 0.096 |
| qwen3_8b_population_prior_s43 | coin_city | arbitrary | original | 1.091 | 0.000 | 0.000 | 0.933 | 0.024 |
| qwen3_8b_population_prior_s43 | coin_city | semantic | oracle_both_patterns | 4.367 | 0.000 | 0.396 | 3.799 | 1.564 |
| qwen3_8b_population_prior_s43 | coin_city | semantic | oracle_selected_pattern | 1.252 | 0.000 | 0.000 | 1.004 | 0.101 |
| qwen3_8b_population_prior_s43 | coin_city | semantic | original | 1.015 | 0.000 | 0.000 | 0.847 | -0.006 |
| qwen3_8b_population_prior_s43 | coin_harbor | arbitrary | oracle_both_patterns | 5.974 | 0.000 | 0.812 | 4.779 | 0.254 |
| qwen3_8b_population_prior_s43 | coin_harbor | arbitrary | oracle_selected_pattern | 8.712 | 0.000 | 0.438 | 8.712 | 0.049 |
| qwen3_8b_population_prior_s43 | coin_harbor | arbitrary | original | 3.011 | 0.000 | 0.000 | 2.464 | 0.063 |
| qwen3_8b_population_prior_s43 | coin_harbor | semantic | oracle_both_patterns | 6.531 | 0.021 | 0.833 | 7.921 | -0.270 |
| qwen3_8b_population_prior_s43 | coin_harbor | semantic | oracle_selected_pattern | 8.879 | 0.000 | 0.312 | 11.027 | 1.132 |
| qwen3_8b_population_prior_s43 | coin_harbor | semantic | original | 2.980 | 0.000 | 0.000 | 2.427 | 0.089 |
| qwen3_8b_population_prior_s44 | coin_city | arbitrary | oracle_both_patterns | 5.959 | 0.000 | 0.708 | 4.923 | 0.135 |
| qwen3_8b_population_prior_s44 | coin_city | arbitrary | oracle_selected_pattern | 1.315 | 0.000 | 0.000 | 1.052 | 0.140 |
| qwen3_8b_population_prior_s44 | coin_city | arbitrary | original | 1.068 | 0.000 | 0.000 | 0.909 | 0.047 |
| qwen3_8b_population_prior_s44 | coin_city | semantic | oracle_both_patterns | 4.815 | 0.021 | 0.312 | 4.196 | 3.537 |
| qwen3_8b_population_prior_s44 | coin_city | semantic | oracle_selected_pattern | 1.258 | 0.000 | 0.000 | 1.006 | 0.070 |
| qwen3_8b_population_prior_s44 | coin_city | semantic | original | 0.964 | 0.000 | 0.000 | 0.797 | -0.002 |
| qwen3_8b_population_prior_s44 | coin_harbor | arbitrary | oracle_both_patterns | 12.535 | 0.021 | 0.708 | 10.028 | 0.862 |
| qwen3_8b_population_prior_s44 | coin_harbor | arbitrary | oracle_selected_pattern | 8.950 | 0.000 | 0.250 | 7.160 | -0.380 |
| qwen3_8b_population_prior_s44 | coin_harbor | arbitrary | original | 2.945 | 0.000 | 0.000 | 2.397 | 0.029 |
| qwen3_8b_population_prior_s44 | coin_harbor | semantic | oracle_both_patterns | 11.196 | 0.062 | 0.833 | 12.827 | -0.095 |
| qwen3_8b_population_prior_s44 | coin_harbor | semantic | oracle_selected_pattern | 6.151 | 0.000 | 0.188 | 6.143 | 1.255 |
| qwen3_8b_population_prior_s44 | coin_harbor | semantic | original | 2.883 | 0.000 | 0.000 | 2.382 | 0.045 |

Baseline omitted means both zero-shock predictions lie within 0.01 of zero, while true resting levels are 50 (Coin City) and 180 (Coin Harbor). JSON includes signed/absolute baseline offsets, per-coordinate relative accuracy, raw forecast MAE, horizon-specific correction, and paired-bootstrap calibration intervals. Every completed model/interface is reported.
