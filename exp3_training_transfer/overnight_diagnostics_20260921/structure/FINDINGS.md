# Structure-selection diagnostic findings

Updated 2026-09-22T03:46:16.259227+00:00; 8/8 frozen endpoints complete.

Cue-to-reference identification is intact: selection-only accuracy ranges from 100.0% to 100.0% across completed models and all four domain/label cells. That identifies a compositional forecasting problem beyond simply matching the cue to a reference. The oracle interfaces do not establish that fourteen noisy observations can yield exact latent coefficients; they isolate use when coefficients are supplied.

## Original task: all three trained seeds

Each trained entry below is an unweighted average of seeds 42, 43, and 44, only shown when all three are available. Calibration 0 means no paired cue response; 1 is the simulator-predicted change. These are descriptive three-seed diagnostics, not a trained-arm significance claim.

| Domain | Labels | Base 8B calibration | Matched calibration (range) | Prior calibration (range) | Matched response MAE | Prior response MAE |
|---|---|---:|---:|---:|---:|---:|
| coin_city | semantic | -0.211 | 0.000 (-0.041 to 0.036) | -0.008 (-0.014 to -0.002) | 1.053 | 0.990 |
| coin_city | arbitrary | 0.344 | 0.017 (0.012 to 0.023) | 0.027 (0.009 to 0.047) | 1.130 | 1.084 |
| coin_harbor | semantic | 0.170 | 0.136 (0.053 to 0.180) | 0.085 (0.045 to 0.122) | 3.318 | 2.903 |
| coin_harbor | arbitrary | -0.094 | 0.008 (-0.068 to 0.067) | 0.035 (0.014 to 0.063) | 3.418 | 3.015 |

## Oracle assistance in the trained models

| Domain | Labels | Arm | Original MAE | Both patterns MAE | Selected pattern MAE | Selected calibration |
|---|---|---|---:|---:|---:|---:|
| coin_city | semantic | causal | 1.053 | 2.213 | 1.440 | -0.034 |
| coin_city | semantic | population_prior | 0.990 | 3.657 | 1.224 | 0.111 |
| coin_city | arbitrary | causal | 1.130 | 3.793 | 1.567 | 0.039 |
| coin_city | arbitrary | population_prior | 1.084 | 4.156 | 1.342 | 0.092 |
| coin_harbor | semantic | causal | 3.318 | 9.657 | 8.633 | -0.969 |
| coin_harbor | semantic | population_prior | 2.903 | 7.938 | 7.197 | 0.218 |
| coin_harbor | arbitrary | causal | 3.418 | 13.837 | 9.303 | -1.070 |
| coin_harbor | arbitrary | population_prior | 3.015 | 10.971 | 8.224 | -0.652 |

## Stronger-model numerical ability and baseline omission

The stronger model frequently returns changes relative to rest instead of absolute forecasts. Clipping those changes to the valid outcome range before computing response differences distorts the response vector, especially in Coin Harbor where the lower bound is 40. The frozen clipped metrics remain the primary results. The following sensitivity was added after auditing raw outputs and is explicitly post hoc. It assesses whether the numerical response is encoded in an incorrectly offset answer; it does not count that answer as a correct forecast.

| Domain | Labels | Original raw response MAE | Both-pattern raw response MAE | Selected raw response MAE | Selected exact relative vectors | Selected baseline omitted | Selected offset-corrected level MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| coin_city | semantic | 4.794 | 0.883 | 0.384 | 0.521 | 0.958 | 0.307 |
| coin_city | arbitrary | 2.721 | 1.225 | 0.525 | 0.417 | 0.917 | 0.420 |
| coin_harbor | semantic | 10.754 | 2.910 | 3.187 | 0.188 | 0.854 | 2.550 |
| coin_harbor | arbitrary | 7.652 | 1.500 | 2.494 | 0.354 | 0.854 | 1.995 |

Exact relative vectors have all eight unclipped response coordinates within 0.01 of truth. Offset correction adds one common shift estimated from the two zero-shock predictions; this is an interpretive check. The 8B oracle failures remain predominantly response-pattern errors after this correction, rather than simple omission of rest.

The evidence supports conditional numerical ability in the stronger model and a failure to combine reference identification, response extraction/use, and absolute forecast output reliably. It does not establish that the original task is solved or that this model family cannot learn it. The full original task remains poor under arbitrary labels and shows inconsistent domain behavior. A separately disclosed supervised pilot can test whether direct forecast supervision learns the missing composition; reference-selection supervision is unnecessary for the already-perfect selection-only subtask.

## Records

The main table and all cell CIs are in REPORT.md and results/*.summary.json. The raw-output audit is in BASELINE_AUDIT.md and baseline_omission_audit.json. The prepared arithmetic prompt followup was not launched because its frozen gate was closed by stronger-model oracle improvement; arithmetic_followup/gate.json preserves that decision. All eight original endpoints remain scheduled independent of outcome. Zero target observations isolate this decomposition; integration with target evidence at k>0 is not tested.
