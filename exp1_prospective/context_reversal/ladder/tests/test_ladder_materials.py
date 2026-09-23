"""The ladder must change difficulty without changing what the analysis rests on."""
import re
from collections import Counter

import pytest

from exp1_prospective.context_reversal.fresh import materials as base
from exp1_prospective.context_reversal.ladder import materials as ladder


@pytest.fixture(scope='module')
def units():
    families = [base.make_family(k, j) for k in range(len(base.CLASSES)) for j in range(2)]
    return [ladder.unit(f, rung, c) for f in families for rung in ladder.RUNGS for c in base.CONTEXTS]


def triples(units):
    groups = {}
    for u in units:
        groups.setdefault((u['parent_family_id'], u['variant']), []).append(u)
    return groups


def test_every_rung_produces_complete_triples(units):
    groups = triples(units)
    assert len(groups) == 16 * len(ladder.RUNGS)
    for key, members in groups.items():
        assert len(members) == 3, key
        assert sorted(m['expected_sign'] for m in members) == [-1, 0, 1], key


def test_triple_holds_evidence_prior_and_actor_identical(units):
    for key, members in triples(units).items():
        assert len({m['messages']['new_news'] for m in members}) == 1, key
        assert len({m['oracle_baseline'] for m in members}) == 1, key
        assert len({m['reported_entity'] for m in members}) == 1, key


def test_triple_has_three_distinct_correct_answers(units):
    """Without this the relation-blind bound of 1/3 and 0 does not hold."""
    for key, members in triples(units).items():
        assert len({m['oracle_update'] for m in members}) == 3, key


def test_word_inventory_is_identical_across_a_triple(units):
    """Extra indirection layers must not give a bag-of-words reader a signal."""
    for key, members in triples(units).items():
        inventories = [Counter(re.findall(r"[A-Za-z']+", m['scenario_text'].lower())) for m in members]
        assert inventories[0] == inventories[1] == inventories[2], key


def test_arithmetic_is_unchanged_across_rungs(units):
    """Same family and context must carry the same oracles at every rung."""
    by_family_context = {}
    for u in units:
        by_family_context.setdefault((u['parent_family_id'], u['context_id']), set()).add(
            (u['oracle_baseline'], u['oracle_update']))
    for key, oracles in by_family_context.items():
        assert len(oracles) == 1, key


def test_rungs_are_strictly_longer_and_deeper(units):
    """Each rung adds binding layers rather than restating the same one."""
    by_rung = {}
    for u in units:
        block = u['scenario_text'].split('ASSIGNMENTS\n')[1].split('\nALREADY KNOWN')[0]
        by_rung.setdefault(u['variant'], []).append(block)
    lengths = {r: sum(len(b) for b in bs) / len(bs) for r, bs in by_rung.items()}
    assert lengths['L0_direct'] < lengths['L1_permit'] < lengths['L2_office'] < lengths['L3_office_distractor']


def test_direct_rung_reproduces_the_frozen_cohort_binding(units):
    """L0 must be the form the frozen cohort already used."""
    l0 = [u for u in units if u['variant'] == 'L0_direct']
    for u in l0:
        block = u['scenario_text'].split('ASSIGNMENTS\n')[1].split('\nALREADY KNOWN')[0]
        assert re.fullmatch(r'(\w+ operates the \w+ channel\. ?){3}', block), block


def test_distractor_names_a_channel_the_actor_does_not_operate(units):
    """The distractor must be inert: it never names the operated channel."""
    for u in (x for x in units if x['variant'] == 'L3_office_distractor'):
        block = u['scenario_text'].split('ASSIGNMENTS\n')[1]
        chairs = {a: o for a, o in re.findall(r'(\w+) chairs the ([\w ]+?)\.', block) if a != 'whoever'}
        offices = dict(re.findall(r'The ([\w ]+?) is held by whoever chairs the ([\w ]+?)\.', block))
        permits = dict(re.findall(r'The (\w+) channel is operated by the holder of the ([\w ]+?)\.', block))
        permit_to_channel = {v: k for k, v in permits.items()}
        office_to_channel = {office: permit_to_channel[permit] for permit, office in offices.items()}
        for actor, office in chairs.items():
            operated = office_to_channel[office]
            named = re.search(rf'{actor} countersigns the log of the (\w+) channel', block)
            assert named and named.group(1) != operated, (u['trial_id'], actor)
