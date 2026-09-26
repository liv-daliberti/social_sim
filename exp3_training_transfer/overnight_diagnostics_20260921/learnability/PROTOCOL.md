# Conditional supervised learnability diagnostic

Prepared 2026-09-22 UTC before component-diagnostic outputs. This is exploratory
mechanism diagnosis, not a new confirmatory seed-level effectiveness claim.

The decision to launch is recorded in `gate.json` with the diagnostic evidence.
Launch forecast SFT if original paired cue-change calibration is below 0.25
for the seed-42 matched RL endpoint in Coin City/semantic, while either oracle
assistance reduces its response MAE by at least 25% or the stronger base reaches
paired calibration above 0.5. These are usability/learnability gates, not
significance tests. If both forms of assistance fail, fix the interface before
training. Selection supervision is considered separately only if selection-only
accuracy in the in-distribution condition is below 90%.

Two fixed forecast-SFT configurations use the same Qwen3-8B base revision,
LoRA rank 32/alpha 64/all seven projection modules, and 4,800 original Coin City
semantic training prompts with simulator-correct ten-number forecasts. One
uses the RL learning rate (1e-6), and a separate optimization-feasibility
condition uses 2e-5. No hyperparameter is selected using evaluation results.
Both start from base, seed 42, one epoch, effective batch 16, constant learning
rate, AdamW (0.9,0.95), no weight decay, maximum gradient norm 1.0. Loss is
completion-only mean token cross entropy per example; prompt tokens are masked.
No truncation is allowed. The only endpoint is after the full 4,800 examples.

This matches distinct prompts and presentations, not compute or optimizer
updates: each SFT run has 300 updates; existing RL has eight sampled completions
per prompt and approximately 2,400 updates across its 300 rollout rounds. The
2e-5 condition also changes learning rate. Therefore results speak to these
specified recipes and budgets, not an isolated universal effect of SFT vs RL.
A failure of one-pass SFT is not evidence of unlearnability. A success with the
higher rate alone makes optimization settings part of the explanation.

Evaluate both endpoints on the same frozen four-interface paired diagnostic as
base/RL/stronger base. Preserve both domains and both label conditions; report
all cells. No arbitrary-label or Coin Harbor training data is introduced.
Also evaluate the original registered held-out set, greedy once and five draws
at temperature 0.7, if time permits; this is a disclosed post-diagnostic test,
never a replacement for the original registered RL analysis.

Neither selecting the better learning rate nor one training seed establishes a
population-level SFT advantage. A positive pilot motivates a separately frozen
replication, with no additional campaign automatically triggered here.

Operational amendment before launch: the scheduler makes short A6000 allocations
available sooner. Training may run in at most six one-hour allocations, pausing
cooperatively after40minutes of optimizer work. LoRA weights, optimizer state,
RNG state and completed-update count are saved; the next allocation resumes the
same fixed example order and skips completed batches. No intermediate evaluation
or endpoint selection occurs. Final component inference runs separately. Original
held-out stochastic evaluation, if run, is a separate additional job.


## Outcome-informed launch amendment (2026-09-22 UTC)

The initial numerical launch rule was not met; its closed result is preserved
in `gate_initial_closed.json`. Before any SFT was submitted, the component
results nevertheless supplied a different reason for the user's explicitly
adaptive learnability pilot. Base8B and matched seed42 identify the named
reference perfectly. The stronger32B model uses supplied selected relationships
substantially better, but raw-output inspection shows frequent return of response
deviations without the resting baseline. The frozen scorer clips out-of-range
forecasts before computing response contrasts, so a separately labeled raw/offset
audit is necessary to distinguish numerical structure from output semantics.

On this observed evidence we launch the two already prepared, unchanged forecast
SFT configurations as a diagnostic of cue-conditioned forecasting and absolute
output semantics. This is an outcome-informed exploratory decision, NOT a passed
predeclared gate or confirmation of full-task feasibility. It adds no training
conditions or seeds, does not select the better learning rate, and does not alter
the frozen evaluation. Both outcomes will be reported, including failure.
Selection-only SFT is not warranted by100% identification accuracy.

The six one-hour allocation limit, one epoch,4800 examples and300 final updates
remain fixed. Frozen clipped scores and clearly labeled raw-response/offset
sensitivities are reported together for all compared models. A positive result
is limited to these specific recipes and this single supervised training seed.
