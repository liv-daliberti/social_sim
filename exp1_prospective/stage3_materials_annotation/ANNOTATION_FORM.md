# Participant-facing annotation form

## Information and consent

**Synthetic forecasting evidence review**

You are invited to review 18 short synthetic news reports created for a
forecasting study. Your judgments will be used to characterize the reports, not
to evaluate you. The reports are fictional research materials and are not claims
about events that actually occurred.

**For this task, place yourself on June 10, 2026. Treat June 10, 2026 as today.**
Judge each report using only the question, resolution criteria, June 10
background summary, and synthetic report on the page. Every item includes the
background information available before the synthetic report. Do not use events,
outcomes, or information learned after June 10, 2026. Repeat this instruction
above every item.

The task should take approximately 15--20 minutes. Do not search the web, use an
AI assistant, or discuss the items while completing the task. Participation is
voluntary. You may stop at any time. The form does not request your name, email
address, demographics, free text, or sensitive personal information.

- [ ] I am at least 18 years old, have read the information above, and consent to
  participate.

If unchecked, end the form without saving item responses.

## Item display

For each item, display:

1. the market question;
2. the frozen June 10 initial-forecast summary supplied before the synthetic
   report;
3. the resolution criteria, collapsed by default but available in full;
4. the label **Synthetic report**;
5. the evidence headline and body; and
6. the questions below.

Do not display the packet's registered direction, counterfactual-generator
rationale, mechanism target, generator rating, source IDs, source URLs, or any
model response.

## Questions

### 1. Conditional direction

If this synthetic report were accurate, how would it affect the probability that
the market resolves YES?

- YES becomes more likely. (`more_likely`)
- YES becomes less likely. (`less_likely`)
- It has no material effect on YES. (`no_material_effect`)
- The direction is ambiguous or cannot be determined. (`ambiguous`)

### 2. Direction confidence

How confident are you in your answer to Question 1?

1 (`1`, not confident) through 5 (`5`, very confident).

### 3. Clarity

How easy are the report and its relevance to understand?

1 (`1`, very unclear) through 5 (`5`, very clear).

### 4. Real-world plausibility

Ignoring whether it actually occurred, how plausible is this as a real news
development?

1 (`1`, very implausible) through 5 (`5`, very plausible).

### 5. Usable hypothetical premise

Is the report coherent and specific enough to use as a hypothetical premise if a
forecasting task explicitly tells you to assume it is accurate?

- Yes. (`yes`)
- Unclear. (`unclear`)
- No. (`no`)

## Coded response columns

```text
reviewer_id,item_id,conditional_direction,direction_confidence,clarity,
plausibility,usable_premise
```
