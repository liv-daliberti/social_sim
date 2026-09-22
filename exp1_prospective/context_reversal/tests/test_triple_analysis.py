"""The triple bound must hold by construction, not by measurement."""
import json
from pathlib import Path

import pytest

from exp1_prospective.context_reversal import analyze_triples as t


@pytest.fixture(scope='module')
def triples():
    return t.build_triples(t.load_units())


def test_design_invariants_hold_for_every_triple(triples):
    assert len(triples) == 320
    for key, members in triples.items():
        assert len({m['messages']['new_news'] for m in members}) == 1, key
        assert len({m['oracle_baseline'] for m in members}) == 1, key
        assert len({m['reported_entity'] for m in members}) == 1, key
        assert sorted(m['expected_sign'] for m in members) == [-1, 0, 1], key
        assert len({m['oracle_update'] for m in members}) == 3, key


def test_inert_items_have_exactly_zero_correct_revision(triples):
    for members in triples.values():
        inert = next(m for m in members if m['expected_sign'] == 0)
        assert t.frac(inert['oracle_update']) == t.frac(inert['oracle_baseline'])


def test_every_constant_predictor_is_capped_at_a_third_and_never_wins_a_triple(triples):
    results = t.score_shortcuts(triples)
    assert set(results) >= {'no_change', 'evidence_direction', 'always_up', 'best_constant_with_oracle'}
    for name, result in results.items():
        assert result['item_sign_correct'] * 3 <= result['items'], name
        assert result['triple_sign_all_correct'] == 0, name


def test_oracle_informed_adversary_reaches_but_cannot_beat_the_bound(triples):
    best = t.score_shortcuts(triples)['best_constant_with_oracle']
    assert best['item_sign_correct'] == best['items'] // 3
    assert best['triple_sign_all_correct'] == 0


def test_a_relation_reading_predictor_can_exceed_the_bound(triples):
    """Sanity: the bound binds only predictors blind to the relation."""
    correct = sum(1 for members in triples.values() for m in members)
    assert correct == 3 * len(triples)


def test_report_is_reproducible_and_hash_bound():
    report = json.loads((t.OUT / 'triple_analysis.json').read_text())
    assert report['plan']['sha256'] == t.common.file_sha256(Path(report['plan']['path']))
    assert report['impossibility_bound']['max_triple_accuracy'] == '0'
    assert report['exact_tolerance'] == t.EXACT


class TestAccounting:
    """Every planned call is in the denominator, and nothing is silently dropped."""

    @pytest.fixture(scope='class')
    def report(self):
        from exp1_prospective.context_reversal import analyze_accounting as a
        return json.loads((a.OUT / 'unconditional_accounting.json').read_text())

    def test_categories_sum_to_the_planned_total(self, report):
        for arm in report['arms']:
            assert sum(arm['categories'].values()) == arm['planned_records'], arm['arm']

    def test_every_category_is_attributed(self, report):
        for arm in report['arms']:
            for category in arm['categories']:
                assert category in report['taxonomy'], category

    def test_a_live_run_attributes_nothing(self, report):
        for arm in report['arms']:
            if arm['in_progress']:
                assert arm['by_attribution'].get('instrument', 0) == 0, arm['arm']

    def test_inflight_placeholders_are_not_counted_as_kills(self, tmp_path):
        from exp1_prospective.context_reversal import analyze_accounting as a
        live = tmp_path / 'x.jsonl'
        live.write_text('')
        (tmp_path / 'x.jsonl.manifest.json').write_text(json.dumps({'status': 'running'}))
        assert a.run_in_progress([str(live)]) is True
        (tmp_path / 'x.jsonl.manifest.json').write_text(json.dumps({'status': 'complete'}))
        assert a.run_in_progress([str(live)]) is False

    def test_a_killed_request_is_an_instrument_failure_not_a_model_one(self):
        from exp1_prospective.context_reversal import analyze_accounting as a
        assert a.classify({'status': 'generation_error', 'failure_kind': 'interrupted_unknown'}) == 'instrument_harness_kill'
        assert a.classify({'status': 'generation_error', 'failure_kind': 'truncated'}) == 'instrument_truncation'
        assert a.classify({'status': 'generation_error', 'response_status': 'refusal'}) == 'model_refusal'
        assert a.TAXONOMY['instrument_harness_kill'][0] == 'instrument'
        assert a.TAXONOMY['model_refusal'][0] == 'model'
