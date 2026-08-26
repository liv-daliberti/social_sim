# Qwen3 Coin City hidden-state probes

This directory contains an appendix-only mechanistic pilot for Experiment 2. It
asks whether City C's latent news-response slope is linearly decodable from
open-weight model hidden states. The completed six-checkpoint roster contains
Qwen3-4B, Qwen3-8B, Qwen3-14B, dense Qwen3-32B, Qwen2.5-72B-Instruct, and
Llama-3.1-8B-Instruct. The frozen cross-architecture extension uses the identical
task file and sealed split. The analysis
is correlational: decodability does not show that the decoded direction causally
mediates the forecast.

## Toy protocol (qwen3_8b_toy_v1)

The pilot uses 64 frozen Coin City episodes (16 from each
strong_reference_city x target_strong cell), with exact prompts from the frozen
Experiment 2 task files. Every selected episode appears in three matched arms:

- abc_context: the City C cue names the correct reference regime;
- abc_no_context: the City C cue is absent;
- abc_wrong_context: the cue names the opposite regime.

Each arm is evaluated at k=0 and k=4 visible City C cases. Episodes, rather
than prompt rows, are assigned to development or test: 12 episodes per factorial
cell enter development and 4 enter the sealed toy test (48/16 total). Thus no
arm or evidence-depth variant of a test episode can enter probe fitting or
hyperparameter/layer selection.

The frozen primary representation is the hidden state at the final input token
after applying Qwen's chat template with enable_thinking=False. We save this
state at the embedding output and every transformer layer. Mean-pooled embedding
and final-layer states are saved as surface-form sensitivities, not as the
primary layer-wise analysis.

For each evidence depth, probes are fit only on development-set abc_context
prompts. Four-fold cell-balanced development cross-validation selects ridge
strength separately at every layer. The development score also selects a single
layer before the sealed test is read. That fixed probe is then evaluated on
matched held-out correct-, absent-, and wrong-context prompts.

## Targets and controls

Three targets separate the easy cue result from the stronger claim:

1. **Response regime** (target_strong): higher versus lower response.
2. **Latent slope** (target_slope): the exact continuous generator parameter.
3. **Within-regime residual slope**: target_slope - 0.90 for higher-response
   targets and target_slope - 0.25 for lower-response targets. This removes
   the information in the explicit qualitative cue. At k=0, the residual is
   deliberately not identifiable from the intended task evidence; at k=4,
   City C's noisy numerical cases can inform it.

Interpretive controls are:

- embedding-output and mean-pooled embedding probes;
- held-out label-permutation nulls that keep the development-selected layer,
  ridge strength, and fitted probe fixed;
- analytic cue-only and visible-C OLS baselines;
- evaluation under cue removal and cue reversal;
- greedy behavioral forecasts on the same held-out prompts.

A useful representational result would require more than high correct-context
regime accuracy. The stronger pattern is: later-layer improvement over controls,
chance residual-slope decoding at k=0, better residual decoding at k=4, and
matched representation/behavior changes when the cue is removed or reversed.
A result confined to regime decoding at k=0 is evidence that the cue is
represented, not that the exact latent slope is recovered.

## Toy outcome (completed 2026-08-24)

The frozen pilot completed on Qwen3-8B (model commit
`b968826d9c46dd6066d109eabc6255188de91218`). All 384 prompts produced finite
hidden states, and all 96 held-out generations parsed. The sealed test contains
16 episodes, so the following numbers are diagnostic rather than confirmatory.

At the development-selected final-token layer, response regime was decodable
under correct context at both evidence depths (k=0 ROC AUC 0.844; k=4 ROC AUC
1.000). Removing the context reduced AUC to 0.453 and 0.578. Reversing it made
the probe track the stated cue rather than the episode's true regime: true-label
AUC was 0.078 at k=0 and 0.000 at k=4, while reversed-cue-label AUC was 0.922
and 1.000. The 10,000-permutation one-sided p-values on correct-context test AUC
were 0.0089 and 0.0002, respectively.

The continuous slope result does not survive the target that matters for an
exact-relationship claim. Full slope R2 was 0.713 at k=0 and 0.653 at k=4, but
within-regime residual-slope R2 was -0.359 and -0.374 (permutation p=0.502 and
0.735). Moreover, mean-pooled raw token embeddings already gave perfect regime
AUC and approximately 0.99 full-slope R2 while failing on residual slope. Thus
the apparent slope signal is predominantly the lexical strong/weak cue and its
between-regime mean, not recovery of fine-grained numerical evidence.

Behavior did not mirror the decodable cue. The median implied slope was 1.0 in
every context/evidence condition. Correct-context slope R2 was -1.58 at k=0 and
-2.21 at k=4; corresponding poll MAE was 2.60 and 2.69 points. On the same test
episodes, a unit-slope baseline has MAE 2.69, whereas using only the known
regime mean has MAE 0.147. High poll-level R2 (about 0.88) is therefore mostly
the easy starting-poll variation, not successful relationship selection.

The defensible conclusion is narrow: Qwen3-8B represents the explicit context
cue, but this toy provides no evidence that it estimates within-regime slope or
uses the represented cue to produce task-optimal forecasts. A stronger follow-up
should first identify an open model with behavioral context sensitivity, then
use lexically matched counterfactual cues and causal activation patching. The
current result should not be promoted to the main text or described as a causal
mechanism.

## Behavioral capability gate (completed 2026-08-24)

Before scaling the hidden-state analysis, we screened Qwen3-14B, dense
Qwen3-32B, and Qwen2.5-72B-Instruct on exactly the same 96 sealed test prompts
used for Qwen3-8B behavior. All four models used greedy, non-thinking native chat
templates and produced 96/96 parseable forecasts. Qwen3-32B is the clean dense,
same-generation scaling comparison. The 72B checkpoint is a cross-generation
Qwen-family robustness control, because Qwen2.5 and Qwen3 differ in model
generation as well as size.

We summarize the screen as a three-question progression. First, **does the model
react to direct City C evidence?** From k=0 to k=4 in the no-context arm, the
regime-aligned implied-slope shift was 0.119 for 8B (95% paired-bootstrap CI
[-0.094, 0.374]), 0.394 for 14B ([0.152, 0.734]), 0.315 for 32B
([0.017, 0.646]), and 0.246 for 72B ([0.001, 0.529]). The 8B interval crosses
zero; the three larger models change regime separation when target-city evidence
appears.

Second, **can the model select a relationship based on context before seeing any
City C cases?** At k=0, correct context versus no context shifted the
regime-aligned slope by 0.078 for 8B ([-0.012, 0.219]), 0.261 for 14B
([0.134, 0.400]), 0.234 for 32B ([0.111, 0.407]), and 0.096 for 72B
([-0.077, 0.259]). Cue inversion supplies the sharper diagnostic: wrong minus
correct context shifted the aligned slope by -0.140 ([-0.315, 0.007]), -0.380
([-0.597, -0.172]), -0.390 ([-0.554, -0.221]), and -0.433
([-0.674, -0.220]), respectively. The same-generation 14B and 32B models show
the clearest correct/absent/inverted pattern; 72B responds strongly to inversion
but only weakly to correct versus absent context.

Third, **is the selected relationship transferred into the numeric forecast?**
At k=0, wrong versus correct context increased poll MAE by 0.812 points for 8B
([-0.063, 2.023]), 1.662 for 14B ([0.252, 3.155]), 1.965 for 32B
([1.013, 3.083]), and 2.206 for 72B ([0.957, 3.620]). Thus inversion changes
the forecast itself for all three larger models. With four direct City C cases,
they partly override the wrong cue: true-regime AUC in the inverted arm rises
from 0.125 to 0.656 for 14B, 0.094 to 0.750 for 32B, and 0.063 to 0.742 for 72B.

This remains regime-level rather than exact-parameter success. Correct-context
continuous-slope R2 at k=0/k=4 is -1.582/-2.214 for 8B, -0.702/-2.779 for 14B,
0.024/-0.795 for 32B, and -0.894/-2.920 for 72B. The small positive 32B k=0
value is effectively zero and disappears when numerical City C evidence is
added. The behavioral gate therefore supports hidden-state probes of Qwen3-14B
and Qwen3-32B for a context-conditioned response-regime representation, while
retaining residual-slope decoding as the control against relabeling lexical cue
information as recovery of the exact latent relationship. Qwen2.5-72B remains a
version-confounded robustness control, not evidence of monotonic scaling.

Frozen screen artifacts and the 10,000-resample episode-paired contrasts are in
`data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/behavior_screens/size_screen_v2.json`.

## Qwen3-14B mechanistic outcome (completed 2026-08-24)

The qualified follow-up used Qwen3-14B commit
`40c069824f4251a91eefaf281ebe4c544efd3e18`. All 384 prompts produced finite
states with shape `[384, 41, 5120]`, and all 96 held-out generations parsed. The
feature artifact SHA-256 is `0c14369a9eeda0fab0dc28b83e0a03d52663e49a2c68c143880e46bfe944247a`.

Selected final-token states decode the context-assigned relationship. Regime AUC
is 1.000/0.984 at k=0/k=4, and full-slope R2 is 0.898/0.627 (all held-out
permutation p<=0.0003). Removing context makes slope R2 negative; inversion makes
the probe fit the stated cue rather than the true relationship.

The strict control is negative: residual-slope R2 is -0.531/-0.128 (p=0.315/0.450).
Mean-pooled input embeddings already reach full-slope R2 0.988/0.992 because the
explicit cue identifies the regime means, while neither pooling control recovers
the residual. Across 8B and 14B, a coarse cue-selected response family is linearly
decodable, but the exact episode-specific coefficient is not. This was the
pre-extension result; the final appendix reports the full six-checkpoint roster,
including the Qwen2.5-72B cross-generation control.

## Commands

Build the deterministic toy task set on CPU:

~~~bash
.runtime/oat_conda/bin/python \
  exp2_v2/biased_news/mechanistic_probe/build_probe_tasks.py
~~~

Submit Qwen3-8B hidden-state extraction and held-out behavioral generation:

~~~bash
sbatch exp2_v2/biased_news/mechanistic_probe/run_qwen3_8b_toy.sbatch
~~~

Submit the behaviorally qualified Qwen3-14B follow-up:

~~~bash
sbatch exp2_v2/biased_news/mechanistic_probe/run_qwen3_14b_probe.sbatch
~~~

Analyze a completed extraction on CPU:

~~~bash
.runtime/oat_conda/bin/python \
  exp2_v2/biased_news/mechanistic_probe/analyze_probe.py \
  --run-dir exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4/mechanistic_probe/qwen3_14b_probe_v1 \
  --permutations 10000 \
  --study qwen3_14b_probe_v1 \
  --status probe_complete
~~~

Regenerate the fail-closed appendix table and figure:

~~~bash
python paper/generate_coin_city_mechanistic_comparison.py
~~~

Probe outputs are kept under the frozen experiment's `mechanistic_probe/`
subdirectory. The full report is wired through
`paper/experiment2_mechanistic_probe_appendix.tex`; the main body contains only
the compact, qualified six-model summary.

## Cross-architecture extension (completed 2026-08-25)

The registered extension added Qwen3-4B-Instruct-2507 and
Llama-3.1-8B-Instruct. The first supplies the smaller same-release Qwen endpoint;
the second changes both vendor and architecture. Both reused the byte-identical
384-prompt task file, episode-disjoint 48/16 development/test assignment, all
three cue arms, both evidence depths, the same final-token anchor and pooling
controls, and the same 10,000-repeat held-out permutation test. No layer, ridge
strength, target, or test episode changed.

Qwen3-4B correct-context regime AUC is 1.000/1.000 at k=0/k=4 and full-slope
R2 is 0.960/0.885. Its within-regime residual R2 is -0.389/-0.227
(p=0.927/0.614). The completed activation tensor has shape `[384, 37, 2560]`,
model commit `cdbee75f17c01a7cc42f958dc650907174af0554`, and SHA-256
`e191385963d639aa5b8574a9abbeb91b3519aed934ac491f863655631b26dbbc`.

Llama-3.1-8B correct-context regime AUC is 1.000/0.984 and full-slope R2 is
0.913/0.731. Its within-regime residual R2 is -0.126/-0.207
(p=0.380/0.572). The completed activation tensor has shape `[384, 33, 4096]`,
model commit `0e9e39f249a16976918f6564b8830bc894c89659`, and SHA-256
`7f65aeec7a91e404cc60068618fedf25c679abde5d48c2b8477f736670fbc54f`.

The input-embedding control is decisive for scope. Every Qwen checkpoint has
perfect regime AUC and approximately 0.99 full-slope R2 from mean-pooled raw
embeddings; Llama has perfect regime AUC and full-slope R2 0.756/0.827. None
recovers within-regime residual slope. Across all six checkpoints, every
correct-context residual test is non-significant (best R2=0.003, p=0.221).
The cross-architecture extension therefore reproduces the same conclusion: the
states contain a coarse, lexically supplied response family, not a reliably
estimated episode-specific coefficient.

`paper/generate_coin_city_mechanistic_comparison.py` now requires all six exact
manifests, task hashes, checkpoint commits, tensor hashes and dimensions,
complete layer curves, and permutation specifications before generating the
paper assets.
