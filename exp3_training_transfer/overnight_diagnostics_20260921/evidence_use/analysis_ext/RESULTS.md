# Mechanism evidence-use diagnostic

Complete: True. Frozen manifest: `d7eacdf2a32c596e0ef37822f5309361563228fae5b054116b188cd5594a1b63`.

Training seeds present: 42, 43, 44, 45, 46. Five or more paired training seeds.

| Disclosure | Arm | Seed | Mechanisms | Delta MAE | No-change MAE | Tracking slope | Parse pairs |
|---|---|---:|---|---:|---:|---:|---:|
| disclosed | base | None | train | 3.405 | 1.591 | 0.261 | 1.000 |
| disclosed | base | None | test | 2.629 | 0.826 | 0.119 | 1.000 |
| disclosed | causal_family | 42 | train | 1.253 | 1.591 | 0.344 | 1.000 |
| disclosed | causal_family | 42 | test | 0.926 | 0.826 | 0.126 | 1.000 |
| disclosed | causal_family | 43 | train | 1.301 | 1.591 | 0.261 | 1.000 |
| disclosed | causal_family | 43 | test | 0.864 | 0.826 | 0.110 | 1.000 |
| disclosed | causal_family | 44 | train | 1.228 | 1.591 | 0.337 | 1.000 |
| disclosed | causal_family | 44 | test | 0.859 | 0.826 | 0.161 | 1.000 |
| disclosed | causal_family | 45 | train | 1.286 | 1.591 | 0.271 | 1.000 |
| disclosed | causal_family | 45 | test | 0.877 | 0.826 | 0.110 | 1.000 |
| disclosed | causal_family | 46 | train | 1.179 | 1.591 | 0.401 | 1.000 |
| disclosed | causal_family | 46 | test | 0.962 | 0.826 | 0.105 | 1.000 |
| disclosed | shuffled_target | 42 | train | 1.547 | 1.591 | 0.049 | 1.000 |
| disclosed | shuffled_target | 42 | test | 0.878 | 0.826 | 0.034 | 1.000 |
| disclosed | shuffled_target | 43 | train | 1.525 | 1.591 | 0.080 | 1.000 |
| disclosed | shuffled_target | 43 | test | 0.897 | 0.826 | 0.040 | 1.000 |
| disclosed | shuffled_target | 44 | train | 1.480 | 1.591 | 0.119 | 1.000 |
| disclosed | shuffled_target | 44 | test | 0.870 | 0.826 | 0.056 | 1.000 |
| disclosed | shuffled_target | 45 | train | 1.492 | 1.591 | 0.089 | 1.000 |
| disclosed | shuffled_target | 45 | test | 0.873 | 0.826 | 0.051 | 1.000 |
| disclosed | shuffled_target | 46 | train | 1.515 | 1.591 | 0.084 | 1.000 |
| disclosed | shuffled_target | 46 | test | 0.852 | 0.826 | 0.065 | 1.000 |
| undisclosed | base | None | train | 3.691 | 1.591 | 0.197 | 1.000 |
| undisclosed | base | None | test | 3.664 | 0.826 | 0.365 | 1.000 |
| undisclosed | causal_family | 42 | train | 1.130 | 1.591 | 0.453 | 1.000 |
| undisclosed | causal_family | 42 | test | 0.864 | 0.826 | 0.256 | 1.000 |
| undisclosed | causal_family | 43 | train | 1.128 | 1.591 | 0.485 | 1.000 |
| undisclosed | causal_family | 43 | test | 0.835 | 0.826 | 0.238 | 1.000 |
| undisclosed | causal_family | 44 | train | 1.038 | 1.591 | 0.529 | 1.000 |
| undisclosed | causal_family | 44 | test | 0.972 | 0.826 | 0.225 | 1.000 |
| undisclosed | causal_family | 45 | train | 1.149 | 1.591 | 0.393 | 1.000 |
| undisclosed | causal_family | 45 | test | 0.827 | 0.826 | 0.220 | 1.000 |
| undisclosed | causal_family | 46 | train | 1.129 | 1.591 | 0.443 | 1.000 |
| undisclosed | causal_family | 46 | test | 0.890 | 0.826 | 0.250 | 1.000 |
| undisclosed | shuffled_target | 42 | train | 1.526 | 1.591 | 0.082 | 1.000 |
| undisclosed | shuffled_target | 42 | test | 0.848 | 0.826 | 0.053 | 1.000 |
| undisclosed | shuffled_target | 43 | train | 1.450 | 1.591 | 0.116 | 1.000 |
| undisclosed | shuffled_target | 43 | test | 0.857 | 0.826 | 0.076 | 1.000 |
| undisclosed | shuffled_target | 44 | train | 1.495 | 1.591 | 0.102 | 1.000 |
| undisclosed | shuffled_target | 44 | test | 0.906 | 0.826 | 0.015 | 1.000 |
| undisclosed | shuffled_target | 45 | train | 1.457 | 1.591 | 0.147 | 1.000 |
| undisclosed | shuffled_target | 45 | test | 0.844 | 0.826 | 0.093 | 1.000 |
| undisclosed | shuffled_target | 46 | train | 1.513 | 1.591 | 0.091 | 1.000 |
| undisclosed | shuffled_target | 46 | test | 0.854 | 0.826 | 0.062 | 1.000 |

The tracking slope is 1 for the simulator and 0 for a model with no response to changed evidence. Delta MAE measures high-minus-low response changes after subtracting the zero-shock forecast.

| Disclosure | Mechanisms | Comparison (positive favors matched) | Effect | 95% interval |
|---|---|---|---:|---|
| disclosed | all | shuffled_target_minus_causal_family_change_mae | 0.167 | [0.139, 0.195] |
| disclosed | all | base_minus_causal_family_change_mae | 2.014 | [1.885, 2.151] |
| disclosed | train | shuffled_target_minus_causal_family_change_mae | 0.262 | [0.212, 0.319] |
| disclosed | train | base_minus_causal_family_change_mae | 2.155 | [2.001, 2.314] |
| disclosed | test | shuffled_target_minus_causal_family_change_mae | -0.024 | [-0.079, 0.027] |
| disclosed | test | base_minus_causal_family_change_mae | 1.731 | [1.493, 1.988] |
| undisclosed | all | shuffled_target_minus_causal_family_change_mae | 0.243 | [0.211, 0.276] |
| undisclosed | all | base_minus_causal_family_change_mae | 2.646 | [2.490, 2.802] |
| undisclosed | train | shuffled_target_minus_causal_family_change_mae | 0.373 | [0.318, 0.426] |
| undisclosed | train | base_minus_causal_family_change_mae | 2.576 | [2.401, 2.758] |
| undisclosed | test | shuffled_target_minus_causal_family_change_mae | -0.016 | [-0.061, 0.027] |
| undisclosed | test | base_minus_causal_family_change_mae | 2.786 | [2.489, 3.082] |
