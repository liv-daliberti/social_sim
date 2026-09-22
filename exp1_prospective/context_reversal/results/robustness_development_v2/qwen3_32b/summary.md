# Controlled-edit robustness development

Model: qwen3_32b; 20 parent families, 4 variants. Complete: True; all valid: True.

| Task | Planned | Present | Valid | Missing | Invalid present |
|---|---:|---:|---:|---:|---:|
| probability | 1280 | 1280 | 1280 | 0 | 0 |
| direction | 960 | 960 | 960 | 0 | 0 |

| Variant | Numeric sign | Numeric paired reversal | Direction sign | Direction paired reversal | Broken unchanged |
|---|---:|---:|---:|---:|---:|
| repaired_base | 22/40 | 7/20 | 28/40 | 8/20 | 2/20 |
| name_only | 24/40 | 6/20 | 27/40 | 9/20 | 3/20 |
| paraphrase | 23/40 | 4/20 | 24/40 | 5/20 | 2/20 |
| resample | 22/40 | 5/20 | 28/40 | 9/20 | 2/20 |

| Edit vs base | New-news update within 2 pp | Mean absolute update difference (pp) | Observed/planned differences | Direction agreement | Both direction judgments correct |
|---|---:|---:|---:|---:|---:|
| name_only | 38/80 | 11.188 | 80/80 | 69/80 | 26/60 |
| paraphrase | 26/80 | 14.312 | 80/80 | 62/80 | 24/60 |
| resample | 42/80 | 11.488 | 80/80 | 72/80 | 28/60 |

Agreement includes masked new-news judgments; joint correctness excludes masked new news, which has no assigned target. Missing and invalid pairs remain failures in both planned binary denominators.

| Variant | Control message | Numeric within 2 pp | Mean absolute movement (pp) | Observed/planned movements | Direction unchanged |
|---|---|---:|---:|---:|---:|
| repaired_base | no_news | 80/80 | 0.000 | 80/80 | 80/80 |
| repaired_base | repeated_news | 78/80 | 0.188 | 80/80 | 75/80 |
| name_only | no_news | 80/80 | 0.000 | 80/80 | 80/80 |
| name_only | repeated_news | 73/80 | 0.875 | 80/80 | 74/80 |
| paraphrase | no_news | 80/80 | 0.000 | 80/80 | 80/80 |
| paraphrase | repeated_news | 77/80 | 0.312 | 80/80 | 73/80 |
| resample | no_news | 80/80 | 0.000 | 80/80 | 80/80 |
| resample | repeated_news | 75/80 | 0.750 | 80/80 | 73/80 |

All planned binary outcomes retain missing and invalid records as failures; magnitude summaries never impute missing values.
Invariance alone is not correctness: agreement and joint correctness are reported separately.
Identical-text resample uses another request seed; excess discrepancy compares edits with this sampling reference.
All variants cluster within the original parent families; no confirmatory or causal identification claim.
Repaired material outcomes do not overwrite or replace original-pilot outcomes.

Full JSON includes intervals, resampling-adjusted discrepancies, missingness and trial-level values.
