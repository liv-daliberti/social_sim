# Is the supervised advantage an effect of SFT, or of rate and budget?

## The confound

Experiment 3 reports that a supervised recipe reaches 97.9-100.0% structure
accuracy in Coin City/semantic where matched RL reaches 50.0%. The two differ in
three ways at once:

                        learning rate   optimizer updates   sequences seen
  RL (matched)          1e-6            2400                38,400
  SFT (higher rate)     2e-5             300                 4,800
  SFT (lower rate)      1e-6             300                 4,800

The lower-rate SFT arm reaches only 58.3%, so the successful arm differs from RL
in BOTH rate and budget. "SFT learns what RL does not" is therefore not a claim
the current roster can support, and the paper says so.

## The missing cell

One arm removes both differences at once: supervised training at the RL learning
rate, run to the RL update budget.

  lr 1e-6, 8 epochs over the same 4,800 prompts = 2,400 optimizer updates

Everything else is held to the frozen recipe: LoRA r32/alpha64 on the seven
projection modules, batch 16, constant schedule, completion-only loss, grad-norm
cap 1.0, Qwen3-8B, the unchanged 768-prompt component diagnostic. Three seeds.

## Outcomes and what each would mean

- Reaches high structure accuracy: supervised training learns response-form
  selection at the same rate and budget where RL does not, and the SFT/RL
  contrast is no longer confounded. This is the result that would license the
  claim the paper currently declines to make.
- Stays near the 58.3% of the 300-update lower-rate arm: the budget was not the
  limitation, and the supervised advantage is attributable to the higher
  learning rate rather than to supervised training as such. The paper's present
  hedge becomes the conclusion rather than a caveat.
- Degrades below 58.3%: eight epochs at this rate overfit; the comparison is
  uninformative about SFT versus RL and is reported as such.

All three are reported. This is declared before the runs.

## Status

Exploratory. Not a registered endpoint. It does not replace the four-seed
higher-rate result, which stands as reported.
