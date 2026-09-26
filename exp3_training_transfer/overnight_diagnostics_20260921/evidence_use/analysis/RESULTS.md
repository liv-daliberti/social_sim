# Mechanism evidence-use diagnostic

Complete: True. Frozen manifest: `222169569599c2926ffccfde8e6db97e88e01bad7f253a6691c2bd63376db0af`.

Only two retained training seeds (45, 46) are available; these results are exploratory.

| Disclosure | Arm | Seed | Mechanisms | Delta MAE | No-change MAE | Tracking slope | Parse pairs |
|---|---|---:|---|---:|---:|---:|---:|
| disclosed | base | None | train | 3.405 | 1.591 | 0.261 | 1.000 |
| disclosed | base | None | test | 2.629 | 0.826 | 0.119 | 1.000 |
| disclosed | causal_family | 45 | train | 1.305 | 1.591 | 0.227 | 1.000 |
| disclosed | causal_family | 45 | test | 0.889 | 0.826 | 0.114 | 1.000 |
| disclosed | causal_family | 46 | train | 1.310 | 1.591 | 0.273 | 1.000 |
| disclosed | causal_family | 46 | test | 0.889 | 0.826 | 0.090 | 1.000 |
| disclosed | population_prior | 45 | train | 1.502 | 1.591 | 0.087 | 1.000 |
| disclosed | population_prior | 45 | test | 0.868 | 0.826 | 0.083 | 1.000 |
| disclosed | population_prior | 46 | train | 1.503 | 1.591 | 0.081 | 1.000 |
| disclosed | population_prior | 46 | test | 0.873 | 0.826 | 0.080 | 1.000 |
| undisclosed | base | None | train | 3.691 | 1.591 | 0.193 | 1.000 |
| undisclosed | base | None | test | 3.658 | 0.826 | 0.369 | 1.000 |
| undisclosed | causal_family | 45 | train | 1.196 | 1.591 | 0.354 | 1.000 |
| undisclosed | causal_family | 45 | test | 0.839 | 0.826 | 0.182 | 1.000 |
| undisclosed | causal_family | 46 | train | 1.129 | 1.591 | 0.434 | 1.000 |
| undisclosed | causal_family | 46 | test | 0.841 | 0.826 | 0.285 | 1.000 |
| undisclosed | population_prior | 45 | train | 1.566 | 1.591 | 0.037 | 1.000 |
| undisclosed | population_prior | 45 | test | 0.843 | 0.826 | 0.074 | 1.000 |
| undisclosed | population_prior | 46 | train | 1.532 | 1.591 | 0.049 | 1.000 |
| undisclosed | population_prior | 46 | test | 0.850 | 0.826 | 0.052 | 1.000 |

The tracking slope is 1 for the simulator and 0 for a model with no response to changed evidence. Delta MAE measures high-minus-low response changes after subtracting the zero-shock forecast.

| Disclosure | Mechanisms | Comparison (positive favors matched) | Effect | 95% interval |
|---|---|---|---:|---|
| disclosed | all | population_prior_minus_causal_family_change_mae | 0.124 | [0.089, 0.157] |
| disclosed | all | base_minus_causal_family_change_mae | 1.978 | [1.780, 2.195] |
| disclosed | train | population_prior_minus_causal_family_change_mae | 0.195 | [0.147, 0.244] |
| disclosed | train | base_minus_causal_family_change_mae | 2.097 | [1.854, 2.348] |
| disclosed | test | population_prior_minus_causal_family_change_mae | -0.019 | [-0.059, 0.024] |
| disclosed | test | base_minus_causal_family_change_mae | 1.739 | [1.347, 2.161] |
| undisclosed | all | population_prior_minus_causal_family_change_mae | 0.259 | [0.225, 0.294] |
| undisclosed | all | base_minus_causal_family_change_mae | 2.624 | [2.392, 2.859] |
| undisclosed | train | population_prior_minus_causal_family_change_mae | 0.386 | [0.338, 0.430] |
| undisclosed | train | base_minus_causal_family_change_mae | 2.528 | [2.279, 2.821] |
| undisclosed | test | population_prior_minus_causal_family_change_mae | 0.007 | [-0.039, 0.051] |
| undisclosed | test | base_minus_causal_family_change_mae | 2.818 | [2.367, 3.300] |
