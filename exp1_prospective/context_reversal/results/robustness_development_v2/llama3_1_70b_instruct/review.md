# Independent robustness review: llama3_1_70b_instruct

Technical audit: **PASS**. This is not a scientific pass/fail gate or permission to select a model.

All 20 parent families and four variants remain grouped; numeric coverage is 1,280 records and categorical coverage 960 records. Raw answers, exact prompts, own-context priors, separately serialized one-user-turn categorical chats, deterministic seeds, checkpoint metadata, and frozen source hashes were checked without model inference. Parent-bootstrap point estimates and intervals were recomputed independently.

| Variant | Numeric sign / 40 | Numeric pair / 20 | Direction sign / 40 | Direction pair / 20 | Broken unchanged / 20 | Broken mean absolute (pp) | Blocked signed priors / 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repaired_base | 24 | 8 | 33 | 13 | 8 | 6.46 | 8 |
| name_only | 24 | 9 | 34 | 14 | 6 | 11.75 | 8 |
| paraphrase | 23 | 6 | 32 | 12 | 7 | 8.70 | 8 |
| resample | 25 | 8 | 32 | 12 | 5 | 8.38 | 7 |

| Edit vs repaired base | New-news update discrepancy mean absolute (pp) | Within 2 pp / 80 | Label agreement / 80 | Both labels correct / 60 | Excess discrepancy vs resample (pp; 95% interval) |
|---|---:|---:|---:|---:|---|
| name_only | 7.69 | 43 | 72 | 38 | 1.48; [-1.9508441875, 5.517166249999997] |
| paraphrase | 7.25 | 45 | 69 | 36 | 1.04; [-1.8965093749999997, 4.254821874999998] |
| resample | 6.21 | 53 | 72 | 37 | 0.00; [0.0, 0.0] |

| Variant | No-news absolute drift (pp) | Repeated-news absolute drift (pp) |
|---|---:|---:|
| repaired_base | 0.000 | 0.016 |
| name_only | 0.000 | 0.001 |
| paraphrase | 0.000 | 0.125 |
| resample | 0.000 | 0.000 |

## Numerical instability versus categorical judgments

| Comparison | Update difference over 2 pp / 80 | Of these, categorical labels agree | Of these, categorical labels disagree |
|---|---:|---:|---:|
| name_only | 37 | 34 | 3 |
| paraphrase | 35 | 29 | 6 |
| resample | 27 | 25 | 2 |

These counts separate numerical update instability from changes in the standalone categorical judgment; they do not condition away failures in the primary outcomes. Numerical differences with agreeing labels can include consistently wrong interpretations, which the joint-correctness column exposes. Compare every edit with the identical-text resample before attributing discrepancy to wording.

Broken-link label accuracy and broken numerical movement provide separate evidence about relevance judgment. If categorical judgments also fail while no-news drift is small, ordinary numerical re-elicitation alone does not explain the observed pattern. Endpoint blocking can contribute to signed numeric failures but cannot mechanically force an erroneous standalone label. An excess-discrepancy interval crossing zero establishes neither an edit effect nor equivalence.


## Scientific interpretation limits

- A stable wrong label is agreement, not successful interpretation; the table keeps joint correctness separate and excludes masked new news only from correctness.
- The resample repeats the exact visible text with another request seed. Its nonzero discrepancy is a descriptive sampling reference. A single resample and descriptive family intervals do not isolate a causal naming or paraphrase effect.
- Endpoint-blocked priors remain unconditional numeric failures. Their counts quantify one mechanical restriction, not a reason to remove difficult items. Standalone categorical errors persist independently of numerical endpoint constraints.
- Compare broken-link categorical failures with broken numerical movement and control drift before attributing failures to probability elicitation alone. Agreement can coexist with unstable numerical updates; the JSON retains that cross-tab with observed denominators.
- Repairs and all edit variants are development observations within the original 20 families. They are not 80 independent families, do not replace original results, and cannot validate the proposed fresh cohort. Future freezing criteria concern materials and execution, not these observed success rates.
