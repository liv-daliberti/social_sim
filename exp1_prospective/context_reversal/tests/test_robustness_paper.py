"""Draft-only gates and rendering checks; synthetic three-model fixtures stay in /tmp."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import write_robustness_paper as writer

QWEN = writer.SOURCE / 'results/robustness_development_v2/qwen3_32b/summary.json'
LEXICAL = writer.SOURCE / 'results/robustness_development_v2/material_audit/lexical_audit.json'


def fixture(directory):
    directory = Path(directory)
    template = writer.load(QWEN)
    parents = sorted({r['parent_family_id'] for r in template['trial_rows']})
    protocol = directory / 'protocol.md'; protocol.write_text('Synthetic paper-writer fixture, not real three-model results.\n')
    submission = {'groups': {}, 'protocol_path': str(protocol), 'protocol_sha256': writer.common.file_sha256(protocol),
                  'code_sha256': {'analyze_robustness.py': template['provenance']['analysis_code']['sha256'],
                                  'audit_robustness_lexical.py': writer.load(LEXICAL)['provenance']['code_sha256']}}
    for model in writer.MODELS:
        results = directory / 'results' / model; results.mkdir(parents=True)
        summary = copy.deepcopy(template); summary['model_key'] = model; summary['synthetic_test_fixture'] = True
        path = results / 'summary.json'; path.write_text(json.dumps(summary))
        sources = {spec['path']: spec['sha256'] for key, spec in summary['provenance'].items() if key != 'analysis_code'}
        sources[str(path)] = writer.common.file_sha256(path)
        audit = {'schema_version': 'independent_robustness_audit_v1', 'audit_pass': True, 'errors': {},
                 'model_key': model, 'sampling_families': parents, 'sources_sha256': sources,
                 'audit_code_sha256': writer.common.file_sha256(writer.SOURCE / 'audit_robustness_results.py'), 'synthetic_test_fixture': True}
        path.with_name('independent_audit.json').write_text(json.dumps(audit))
        submission['groups'][model] = {'results_dir': str(results),
                                      **{key: summary['provenance'][key]['path'] for key in ('probability_responses', 'direction_responses')}}
    material = directory / 'results/material_audit'; material.mkdir()
    lexical = writer.load(LEXICAL)
    lexical.pop('context_inventory_checks', None)
    for baseline in lexical['held_out_baselines'].values(): baseline.pop('predictions', None)
    (material / 'lexical_audit.json').write_text(json.dumps(lexical))
    path = directory / 'submission.json'; path.write_text(json.dumps(submission))
    return path, submission


class DraftTests(unittest.TestCase):
    def test_complete_audited_fixture_has_12_9_12_rows_and_separate_controls(self):
        with tempfile.TemporaryDirectory() as d, patch.object(writer, 'RUN', Path(d) / 'run'):
            submission, _ = fixture(d)
            output = writer.RUN / 'paper_draft'
            result = writer.write(submission, output)
            self.assertEqual(result['status'], 'complete_audited_draft_for_root_review')
            expected = {'performance': 12, 'edits': 9, 'controls': 12}
            for name, count in expected.items():
                lines = (output / 'tables' / f'exp1_context_robustness_v2_{name}.tex').read_text().splitlines()
                rows = [line for line in lines if any(line.startswith(model + ' &') for model in writer.MODELS.values())]
                self.assertEqual(len(rows), count)
            prose = (output / 'exp1_context_robustness_v2.tex').read_text()
            self.assertIn('20 sampling families', prose)
            self.assertIn('not a causal estimate', prose)
            self.assertIn('single resample does not isolate', prose)
            self.assertIn('do not establish that all shortcuts are removed', prose)
            self.assertIn('60 determinate pairs', prose)
            self.assertIn('numerical paired reversal is 7/20, 7/20, 7/20', prose)
            self.assertEqual(result['draft_manifest_sha256'], writer.common.file_sha256(output / 'draft_manifest.json'))
            self.assertEqual(len(result['generated_sha256']), 4)
            for relative, digest in result['generated_sha256'].items():
                self.assertEqual(writer.common.file_sha256(output / relative), digest)
            self.assertEqual(writer.write(submission, output), result)
            controls = (output / 'tables/exp1_context_robustness_v2_controls.tex').read_text()
            self.assertIn('No news:', controls); self.assertIn('Repeated:', controls)

    def test_missing_summary_failed_audit_changed_hash_or_bootstrap_produces_no_draft(self):
        for failure in ('missing', 'audit', 'hash', 'bootstrap'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d, patch.object(writer, 'RUN', Path(d) / 'run'):
                submission, spec = fixture(d)
                path = Path(spec['groups']['llama3_1_70b_instruct']['results_dir']) / 'summary.json'
                if failure == 'missing': path.unlink()
                elif failure == 'audit':
                    audit = path.with_name('independent_audit.json'); value = writer.load(audit); value['audit_pass'] = False; audit.write_text(json.dumps(value))
                elif failure == 'hash': path.write_text(path.read_text() + '\n')
                else:
                    value = writer.load(path); value['bootstrap']['unit'] = 'variant_family_id'; path.write_text(json.dumps(value))
                output = writer.RUN / 'paper_draft'
                with self.assertRaises(ValueError): writer.write(submission, output)
                self.assertFalse(output.exists())

    def test_real_partial_cohort_is_refused_without_publishing_qwen_only(self):
        submission = writer.load(writer.RUN / 'submission.json')
        if all((Path(g['results_dir']) / 'summary.json').exists() for g in submission['groups'].values()):
            self.skipTest('Real cohort is now complete; synthetic missing-model test still covers gate')
        with self.assertRaisesRegex(ValueError, 'Completed summary and independent audit required'):
            writer.inputs(writer.RUN / 'submission.json')

    def test_missing_magnitude_not_zero_and_paper_output_forbidden(self):
        metric = {'estimate': None, 'ci95': None, 'n_observed': 0, 'n_planned': 20}
        self.assertEqual(writer.magnitude(metric), '-- (0/20 obs.)')
        with self.assertRaisesRegex(ValueError, 'paper files are never overwritten'):
            writer.write(writer.RUN / 'submission.json', writer.REPO / 'paper')
        summary = writer.load(QWEN)
        summary['by_variant']['repaired_base']['numeric_sign_correct']['n_planned'] = 39
        with self.assertRaisesRegex(ValueError, 'denominator'):
            writer.validate_summary(summary, 'qwen3_32b')


if __name__ == '__main__': unittest.main()
