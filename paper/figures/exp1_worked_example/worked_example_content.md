# Experiment 1 — Worked Example (GPT-5.4)

**Question:** Trump orders federal review of AI model releases by June 30?
**Category:** Tech · **Market price (YES):** 38% · **Days to resolution:** 20.1

## 1. Event model (the causal model GPT-5.4 built)

**Key actors:** Donald Trump; White House policy staff; Department of Commerce; National Institute of Standards and Technology (NIST); National security agencies; Congress; Frontier AI developers such as OpenAI, Anthropic, Google, Meta, and xAI; Polymarket resolvers interpreting the market criteria

**Key mechanisms:**
- Presidential executive action can immediately create or direct a federal AI model review framework
- A qualifying action must target review of public release of new AI models rather than procurement or internal government use
- The June 2, 2026 executive order created a voluntary framework for early federal access to covered frontier models before release
- Resolution likely turns on whether that voluntary pre-release framework counts as a qualifying federal review process under the market rules
- Additional agency guidance or follow-up presidential action before June 30 could clarify whether the framework is review-like enough to qualify

**Latent variables (what's unknown):**
- How Polymarket interprets a voluntary pre-release framework versus mandatory pre-clearance
- Whether follow-up implementation documents before June 30 make the review process more explicit
- Whether the administration intends the June 2 order to function as substantive release oversight or only cooperative national-security screening
- Whether any additional executive action is issued before the deadline
- How strongly the White House's disclaimer against mandatory licensing and pre-clearance weighs against a YES resolution

## 2. Hypotheses (prior → posterior)

- **H1 (YES)** — prior 22% → posterior 46%: The market resolves YES: by June 30, 2026, Trump has already taken, or will still take, executive action or sign legislation that creates, authorizes, or directs a federal review process for the public release of new AI models, with the June 2, 2026 executive order potentially qualifying because it establishes a voluntary pre-release federal access and benchmarking framework for covered frontier models.
- **H2 (NO)** — prior 78% → posterior 54%: The market resolves NO: by June 30, 2026, there is no qualifying federal review process for the public release of new AI models because the June 2, 2026 order is judged too voluntary, too national-security-focused, or too clearly disclaimed as non-preclearance to satisfy the resolution criteria, and no stronger follow-up action occurs before the deadline.

**Mapping:** Since H1=YES and H2=NO, yes_prob = posterior_probability(H1) directly. H1 posterior: 46%, therefore yes_prob = 46%.

**Initial forecast p̂ = 46%** (confidence: medium)

**Initial rationale:** The key update is that Trump already signed a June 2, 2026 executive order creating a voluntary pre-release federal access and benchmarking framework for covered frontier models. The forecast stays slightly below 50% because the same White House fact sheet explicitly disclaims mandatory licensing or pre-clearance, leaving substantial ambiguity about whether the action satisfies the market's requirement for a federal review process for public release.

## 3. Counterfactual update — MAIN [pro_H1]  ·  plausibility 4/5

**Injected headline (June 10, 2026):** OMB tracking figures show agencies racing to finalize AI release review framework

> WASHINGTON, June 10 — Internal Office of Management and Budget tracking figures reviewed by Reuters showed 14 federal agencies had returned comments this week on a draft framework for reviewing the public release of certain advanced AI models, with target deadlines of June 24 for final criteria and June 27 for presidential sign-off. The figures, circulated in an interagency budget and regulatory update, listed Commerce, Defense, Homeland Security and Justice among offices assigned implementation tasks tied to a potential executive action. The unusually compressed timeline and cross-agency staffing matrix suggest the administration is preparing a binding federal process before the end of June.

*Targets mechanism:* Whether a White House-directed executive action creating a federal AI model release review process is already being actively drafted and coordinated across agencies

**Forecast: 46% → 67% (Δ = +0.21; correct direction: True)**

**Updated rationale:** The new evidence materially increases the chance of YES because it goes beyond reinterpreting the June 2 order: it points to a near-complete interagency process for a draft framework explicitly described as reviewing the public release of advanced AI models, with concrete deadlines before June 30 and planned presidential sign-off. That sharply raises the odds of a qualifying executive action, though some residual ambiguity remains because the final action could still be framed as nonbinding or fall short of what resolvers treat as a true federal review process.

## Contrast — opposite direction [anti_H1]  ·  plausibility 4/5

**Injected headline (June 10, 2026):** White House rules out June order creating AI model release reviews

> WASHINGTON, June 10 — The White House said on Wednesday that President Donald Trump will not sign any June directive establishing a federal review or approval process for public releases of new artificial intelligence models, calling such a move "premature and unnecessarily restrictive." Speaking after a meeting with technology executives, White House press secretary Elena Cross said the administration would instead pursue nonbinding guidance on model security and export controls later this summer. The statement appeared to shut the door on industry speculation that Trump was weighing an executive order before June 30.

*Targets mechanism:* An explicit presidential refusal, delivered by the White House, to use executive action before the deadline to create a qualifying federal AI model release review process.

**Forecast: 46% → 27% (Δ = -0.19; correct direction: True)**

**Updated rationale:** The new June 10 White House statement materially reduces the chance of any additional qualifying action before June 30 by explicitly ruling out a June directive establishing a federal review or approval process for public model releases. That shifts weight toward NO, though YES remains possible because the existing June 2 executive order could still be interpreted by resolvers as already having created a sufficient pre-release review framework.

## Contrast — orthogonal (selectivity null) [orthogonal]  ·  plausibility 4/5

**Injected headline (June 10, 2026):** Utilities warn AI data-center surge is straining summer power planning

> ATLANTA, June 10 — U.S. grid operators and electric utilities said on Wednesday that a rapid build-out of AI-focused data centers is complicating summer reliability planning, with several Southeast and Midwest providers seeking faster forecasts of large industrial power demand. At an Edison Electric Institute meeting, utility executives said they were not asking for new federal controls on AI products, but wanted clearer timelines from developers on when major computing sites would come online. Industry analysts said the power scramble highlights how AI expansion is rippling through infrastructure markets even as Washington debates broader technology policy.

*Targets mechanism:* Infrastructure and electricity-demand pressures surrounding AI deployment, rather than presidential or agency action on reviewing public releases of new AI models

**Forecast: 46% → 44% (Δ = -0.02; correct direction: None)**

**Updated rationale:** The new evidence is only weakly relevant to the resolution criteria. It suggests current stakeholder pressure around AI is centered on electricity-demand planning for data centers rather than new federal controls on AI products, and utilities explicitly said they are not seeking such controls. That slightly reduces the chance of additional late-June action that would clarify or strengthen the June 2 order into a more obviously qualifying federal review process, but it does not materially change the ambiguity around whether the existing order already counts.
