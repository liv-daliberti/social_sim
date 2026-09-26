# Post hoc baseline and raw-response sensitivity

Post hoc sensitivity motivated by stronger-model forecasts omitting the resting baseline. Frozen primary metrics unchanged.

This audit subtracts each horizon's predicted zero-shock output before calculating response error, without clipping predictions first. It separates an omitted resting level from incorrect response patterns. Exact relative accuracy means all eight response coordinates are within 0.01. Offset-corrected MAE adds one common shift estimated from the two zero-shock predictions; it is an interpretive sensitivity, not a valid submitted forecast or replacement primary outcome.

| Model | Domain | Labels | Interface | Raw response MAE | Relative exact | Baseline omitted | Offset-corrected MAE | Raw cue calibration |
|---|---|---|---|---:|---:|---:|---:|---:|
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
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | arbitrary | oracle_both_patterns | 1.630 | 0.042 | 0.979 | 1.304 | 0.180 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | arbitrary | oracle_selected_pattern | 1.825 | 0.188 | 1.000 | 1.460 | 1.489 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | arbitrary | original | 1.727 | 0.000 | 0.000 | 1.384 | 0.001 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | semantic | oracle_both_patterns | 2.130 | 0.021 | 0.896 | 1.704 | 1.222 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | semantic | oracle_selected_pattern | 1.859 | 0.146 | 0.688 | 1.488 | 0.674 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_city | semantic | original | 1.253 | 0.000 | 0.000 | 1.016 | 0.166 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | arbitrary | oracle_both_patterns | 3.327 | 0.000 | 1.000 | 2.662 | 0.014 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | arbitrary | oracle_selected_pattern | 3.957 | 0.021 | 1.000 | 3.166 | 0.369 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | arbitrary | original | 5.965 | 0.000 | 0.000 | 4.772 | 0.120 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | semantic | oracle_both_patterns | 12.359 | 0.062 | 0.917 | 9.887 | 1.132 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | semantic | oracle_selected_pattern | 13.049 | 0.125 | 0.896 | 10.440 | 2.814 |
| qwen3_8b_sft_forecast_lr1e6_s42 | coin_harbor | semantic | original | 5.317 | 0.000 | 0.000 | 4.271 | 0.058 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | arbitrary | oracle_both_patterns | 2.253 | 0.000 | 0.000 | 1.802 | -0.015 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | arbitrary | oracle_selected_pattern | 1.981 | 0.000 | 0.000 | 1.611 | 0.273 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | arbitrary | original | 1.274 | 0.000 | 0.000 | 1.019 | -0.211 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | semantic | oracle_both_patterns | 1.669 | 0.000 | 0.000 | 1.358 | 0.474 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | semantic | oracle_selected_pattern | 1.117 | 0.000 | 0.000 | 0.894 | 1.320 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_city | semantic | original | 0.658 | 0.000 | 0.000 | 0.527 | 1.413 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | arbitrary | oracle_both_patterns | 12.865 | 0.000 | 0.021 | 10.631 | 0.381 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | arbitrary | oracle_selected_pattern | 10.955 | 0.000 | 0.000 | 9.149 | 0.320 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | arbitrary | original | 10.655 | 0.000 | 0.000 | 8.839 | 0.004 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | semantic | oracle_both_patterns | 13.702 | 0.000 | 0.000 | 11.270 | -0.097 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | semantic | oracle_selected_pattern | 11.527 | 0.000 | 0.000 | 9.617 | 0.015 |
| qwen3_8b_sft_forecast_lr2e5_s42 | coin_harbor | semantic | original | 12.192 | 0.000 | 0.000 | 10.007 | -0.932 |

Baseline omitted means both zero-shock predictions lie within 0.01 of zero, while true resting levels are 50 (Coin City) and 180 (Coin Harbor). JSON includes signed/absolute baseline offsets, per-coordinate relative accuracy, raw forecast MAE, horizon-specific correction, and paired-bootstrap calibration intervals. Every completed model/interface is reported.
