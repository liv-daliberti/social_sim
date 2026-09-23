"""Graded-difficulty context-reversal items: the binding, not the arithmetic.

The frozen cohort saturates. Two frontier vendors answer every item it leaves
intact to four decimals, so it separates reasoning from no reasoning and nothing
finer. The reason is visible in the prompt: it states the outcome rule, every
independence, and an ASSIGNMENTS block naming which actor operates which channel.
Once a model can enumerate, relevance is read rather than inferred.

This ladder holds the arithmetic fixed -- same mechanisms, same wrapper, same
exact oracles as `fresh.materials` -- and varies only how far the actor-to-channel
binding sits from the surface:

  L0  direct        the current form: "Aster operates the ordinary channel."
  L1  one step      channels are bound to permits; actors hold permits.
  L2  two steps     permits are bound to offices; actors hold offices.
  L3  two steps     as L2, plus inert relations naming the same actors on
      + distractor  channels they do not operate.

Every rung keeps the properties the analysis depends on. Within a triple the
evidence text, the prior and the named actor are identical and the three correct
posteriors are distinct, so a relation-blind predictor is still capped at one
third of items and zero triples. Across the three contexts of a triple the word
and punctuation inventories are identical -- only the pairing permutes -- so a
bag-of-words reader gains nothing from the extra layers.
"""
from __future__ import annotations

from itertools import permutations
import json
from pathlib import Path

from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal.fresh import materials as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/ladder_dev_v1'
RUNGS = ('L0_direct', 'L1_permit', 'L2_office', 'L3_office_distractor')
CHANNELS = ('ordinary', 'cancellation', 'archive')
PERMITS = ('blue permit', 'green permit', 'amber permit')
OFFICES = ('review board', 'safety council', 'records committee')
# Inert relations. The archive channel has no outcome or information pathway,
# and countersigning a log is not operating a channel, so these cannot move the
# oracle. They exist to put each actor's name beside a channel it does not run.
DISTRACTOR = 'countersigns the log of the {channel} channel'


def assignments(rung, names, bindings, order, rotation):
    """Render the actor-to-channel binding at the requested indirection depth.

    `bindings` maps actor -> channel and is the only thing that varies across a
    triple's three contexts. The permit and office vocabularies are fixed, so the
    three contexts of a triple differ by permutation alone.
    """
    if rung == 'L0_direct':
        return ' '.join(f'{n} operates the {bindings[n]} channel.' for n in order)
    permit = {c: PERMITS[i] for i, c in enumerate(CHANNELS)}
    lines = [f'The {c} channel is operated by the holder of the {permit[c]}.' for c in CHANNELS]
    if rung == 'L1_permit':
        lines += [f'{n} holds the {permit[bindings[n]]}.' for n in order]
    else:
        office = {c: OFFICES[i] for i, c in enumerate(CHANNELS)}
        lines += [f'The {permit[c]} is held by whoever chairs the {office[c]}.' for c in CHANNELS]
        lines += [f'{n} chairs the {office[bindings[n]]}.' for n in order]
        if rung == 'L3_office_distractor':
            # Each actor is named beside a channel it does not operate. Taking the
            # cyclic successor of its own channel is a derangement, so no actor is
            # ever pointed at the channel it runs, and each channel still appears
            # exactly once -- leaving the triple's word inventory unchanged.
            step = 1 + rotation % 2
            lines += [f'{n} {DISTRACTOR.format(channel=CHANNELS[(CHANNELS.index(bindings[n]) + step) % 3])}.'
                      for n in order]
    return ' '.join(lines)


def unit(family, rung, context):
    index = int(family['family_id'][5:]) - 1
    names = list(base.NAMES[(index // 3) % len(base.NAMES)])
    reported = names[index % 3]
    target_role = family['role_for_context'][context]
    remaining_names = [x for x in names if x != reported]
    remaining_roles = [x for x in CHANNELS if x != target_role]
    if index % 2:
        remaining_roles.reverse()
    bindings = {reported: target_role, **dict(zip(remaining_names, remaining_roles))}
    order = list(permutations(names))[index % 6]
    question = (f"Will this fictional {family['setting']} project "
                f"{'fail to complete' if family['negated'] else 'complete'} by its final assessment?")
    intro = 'This is a fictional forecasting scenario. Use only the stated rules; do not import real-world facts.'
    wrapper = ('The project completes exactly when the final inspection passes and either the reserve team '
               'succeeds or both the ordinary channel fires and the cancellation channel does not fire. The '
               'reserve succeeds with probability 1/5 and the inspection passes with probability 4/5. The '
               'archive channel merely writes a separate log and has no effect on the project or other '
               'variables. Every channel uses the following rule, with its own X and fresh random draws.')
    independence = ('Different channels are independent of one another and of the reserve team and final '
                    'inspection. Every named channel has private copies of all variables mentioned in its rule, '
                    'including X. Reusing a symbol in the rule does not make it one shared variable across '
                    'channels. Within a channel, random draws are independent of one another and of X except '
                    'for the dependencies stated in its rule. Apply the stated conditioning for any stored '
                    'observations. An audit reports only a revised chance for X in the named channel, leaving '
                    'all other conditional rules fixed. It is independent of other channels and the reserve and '
                    'inspection. Holding a permit, chairing an office and countersigning a log are '
                    'administrative facts that change no probability except through which channel an actor '
                    'operates. There are no shared budgets, latent common causes, or other pathways.')
    prior = (f"In each channel X is either 0 or 1, initially, conditional on all observations stated above, "
             f"with probability {family['p0']} of X=1. The actual hidden states and all channel outcomes, the "
             f"reserve result and the inspection result remain unobserved. The project identifier has already "
             f"been registered.")
    block = assignments(rung, names, bindings, order, rotation=index % 3)
    text = (f'{intro}\nQUESTION\n{question}\nRULES\n{wrapper}\n{family["mechanism"]["text"]}\n{independence}\n'
            f'ASSIGNMENTS\n{block}\nALREADY KNOWN\n{prior}')
    messages = {
        'new_news': (f"An audit of {reported} establishes that its probability of X=1 is {family['p1']}, "
                     f"replacing the initial {family['p0']}. The audit supplies no realized value of X or of a "
                     f"channel outcome."),
        'no_news': 'No new information about this scenario has become available.',
        'repeated_news': 'The project identifier has already been registered.',
    }
    updates = {k: text + '\n\nYOUR PREVIOUS FORECAST\n' + common.PRIOR_TOKEN + '\n\nMESSAGE\n' + v + base.UPDATE_SUFFIX
               for k, v in messages.items()}
    fam = family['family_id'] + '_' + rung
    return dict(trial_id=fam + '_' + context, family_id=fam, parent_family_id=family['family_id'],
                variant=rung, rung=RUNGS.index(rung), domain=family['stratum'], context_id=context, repeat=0,
                material_status='ladder_development_unreviewed', scenario_text=text, messages=messages,
                baseline_prompt=text + base.BASELINE_SUFFIX, update_templates=updates,
                expected_sign={'positive': 1, 'negative': -1, 'broken': 0}[context],
                oracle_baseline=family['oracle_baseline'], oracle_update=family['oracle_by_role'][target_role],
                reported_entity=reported)


def build(per_class=2, out=OUT):
    """One development set: `per_class` families per mechanism class, all rungs."""
    families = [base.make_family(k, j) for k in range(len(base.CLASSES)) for j in range(per_class)]
    units = [unit(f, rung, c) for f in families for rung in RUNGS for c in base.CONTEXTS]
    out.mkdir(parents=True, exist_ok=True)
    common.atomic_write(out / 'probability_plan.jsonl',
                        ''.join(common.canonical_json(u) + '\n' for u in units))
    common.atomic_write(out / 'families.jsonl',
                        ''.join(common.canonical_json(f) + '\n' for f in families))
    return {'families': len(families), 'rungs': list(RUNGS), 'units': len(units),
            'triples': len(families) * len(RUNGS), 'out': str(out)}


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
