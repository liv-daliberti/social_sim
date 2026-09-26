# Local context-reversal diagnostics

Exploratory diagnostics on unchanged development materials. The existing human materials review remains complete; these results do not claim new review coverage.

All five prescribed arms are retained. Missing and invalid responses fail planned-denominator binary metrics; no forecast is imputed. Intervals are descriptive family-bootstrap intervals. Results do not select a model.

Status: **complete_with_errors**. Record complete: **true**. All valid: **false**. Expected responses: **1360**.

Probability sign and paired-reversal columns use the original raw new-news-minus-baseline outcomes. Direction columns use the categorical companion task; their scores measure different responses.

| Task | Model / arm | Present / planned | Valid | Missing | Invalid present | Sign accuracy | Paired reversal | Broken-link diagnostic |
|---|---|---:|---:|---:|---:|---|---|---|
| direction | qwen2_5_72b_instruct | 240 / 240 | 240 | 0 | 0 | 62.5% (25/40) [50.0, 75.0] | 30.0% (6/20) [10.0, 50.0] | unchanged: 30.0% (6/20) [10.0, 50.0] |
| direction | llama3_1_70b_instruct | 240 / 240 | 240 | 0 | 0 | 77.5% (31/40) [67.5, 87.5] | 55.0% (11/20) [35.0, 75.0] | unchanged: 35.0% (7/20) [15.0, 55.0] |
| direction | qwen3_32b | 240 / 240 | 240 | 0 | 0 | 70.0% (28/40) [57.5, 82.5] | 45.0% (9/20) [25.0, 65.1] | unchanged: 10.0% (2/20) [0.0, 25.0] |
| probability | qwen3_32b_unconstrained_nonthinking | 320 / 320 | 320 | 0 | 0 | 62.5% (25/40) [50.0, 75.0] | 30.0% (6/20) [10.0, 50.0] | raw_new_news: criterion_not_met (complete_planned_observations); drift_adjusted: criterion_not_met (complete_planned_observations) |
| probability | qwen3_32b_thinking | 320 / 320 | 300 | 0 | 20 | 75.0% (30/40) [60.0, 87.5] | 60.0% (12/20) [35.0, 80.0] | raw_new_news: indeterminate (missing_planned_observations); drift_adjusted: indeterminate (missing_planned_observations) |

Probability broken-link stability requires complete measurements, the signed-mean 90% interval strictly inside ±2 percentage points, and the upper 95% mean-absolute interval below 2 points. Nonsignificance is not equivalence.

## Response failures and analysis status

Invalid-present counts are totals; individual failure statuses below are their components, not additional records. An analysis error leaves coverage unknown instead of treating corrupt records as absent.

| Task | Model / arm | Analysis succeeded | Failure counts | Analysis error |
|---|---|---|---|---|
| direction | qwen2_5_72b_instruct | true | invalid_present=0; missing_record=0; unclear=0 | none |
| direction | llama3_1_70b_instruct | true | invalid_present=0; missing_record=0; unclear=1 | none |
| direction | qwen3_32b | true | invalid_present=0; missing_record=0; unclear=2 | none |
| probability | qwen3_32b_unconstrained_nonthinking | true | invalid_present=0; missing_record=0 | none |
| probability | qwen3_32b_thinking | true | invalid_present=20; missing_record=0; reason:blocked_baseline=15; reason:parse_error=5 | none |

## Direction control judgments

Masked new messages have no prespecified direction target. Their label counts and unclear rates remain in the per-arm analysis, and all control conditions below retain planned denominators.

| Model | Context | Condition | Accuracy | Unclear rate |
|---|---|---|---|---|
| qwen2_5_72b_instruct | positive | no_news | 90.0% (18/20) [75.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | positive | repeated_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | negative | no_news | 85.0% (17/20) [70.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | negative | repeated_news | 90.0% (18/20) [75.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | broken | no_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | broken | repeated_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | masked | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen2_5_72b_instruct | masked | repeated_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | positive | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | positive | repeated_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | negative | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | negative | repeated_news | 80.0% (16/20) [60.0, 95.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | broken | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | broken | repeated_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | masked | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| llama3_1_70b_instruct | masked | repeated_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | positive | no_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | positive | repeated_news | 90.0% (18/20) [75.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | negative | no_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | negative | repeated_news | 85.0% (17/20) [70.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | broken | no_news | 95.0% (19/20) [85.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | broken | repeated_news | 90.0% (18/20) [75.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | masked | no_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |
| qwen3_32b | masked | repeated_news | 100.0% (20/20) [100.0, 100.0] | 0.0% (0/20) [0.0, 0.0] |

Scheduler accounting is recorded separately from response completeness. A completed scheduler job alone does not establish valid responses. Per-arm summary paths, response hashes, analysis commands, and full metrics are retained in completion.json.
