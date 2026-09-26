"""No-network certificate binding, schema preservation, and variable-prior bounds."""
import copy
import json
import unittest

from exp1_prospective.context_reversal import frontier_token_certificate as certificate
from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal.tests.test_frontier_budget import frozen_fixture
from exp1_prospective.context_reversal.tests.test_runner import make_unit


def fake_receipts(requests, count=200):
    return [{'request_id':r['request_id'],'endpoint':certificate.ENDPOINT,'http_status':200,
             'count_body_sha256':r['count_body_sha256'],'response':{'object':'response.input_tokens','input_tokens':count},
             'provider_request_id':'req_OFFLINE_SYNTHETIC_'+str(i),'received_at':'2026-09-22T00:00:00Z'}
            for i,r in enumerate(requests['requests'])]


class CertificateTests(unittest.TestCase):
    def fixture(self):
        unit=make_unit();unit['baseline_prompt']='A fresh nonempty baseline question'
        requests=certificate.request_manifest([unit],'plan-hash',2048)
        value=certificate.import_receipts(requests,fake_receipts(requests))
        return unit,requests,value

    def test_exact_schema_payload_bound_and_extreme_prior_serialization(self):
        unit,requests,value=self.fixture()
        for request in requests['requests']:
            self.assertEqual(request['count_body']['text'],request['generation_body']['text'])
            self.assertEqual(request['count_body']['reasoning'],{'effort':'low'})
        bounds=certificate.certified_bounds([unit],'plan-hash',2048,value)
        baseline=bounds[common.canonical_json([unit['trial_id'],'baseline','baseline'])]
        self.assertEqual(baseline['input_token_upper_bound'],200)
        for p in (0.0,1.0,.12345678901234567,5e-324,1e-300,0.9999999999999999):
            self.assertLessEqual(len(json.dumps(p).encode()),32)
            for arm in common.CONDITIONS:
                limit=bounds[common.canonical_json([unit['trial_id'],'update',arm])]
                prompt=common.render_update(unit,arm,p)
                self.assertLessEqual(len(prompt.encode()),limit['max_prompt_utf8_bytes'])
                self.assertEqual(limit['input_token_upper_bound'],200+limit['max_prompt_utf8_bytes'])

    def test_missing_duplicated_changed_body_and_schema_evidence_fail_closed(self):
        unit,requests,value=self.fixture()
        receipts=fake_receipts(requests)
        for changed in (receipts[:-1],receipts+receipts[:1]):
            with self.assertRaises(ValueError):certificate.import_receipts(requests,changed)
        changed=copy.deepcopy(receipts);changed[0]['count_body_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'body changed'):certificate.import_receipts(requests,changed)
        changed=copy.deepcopy(unit);changed['baseline_prompt']+=' changed'
        with self.assertRaisesRegex(ValueError,'exact plan/model/schema'):
            certificate.certified_bounds([changed],'plan-hash',2048,value)
        changed=copy.deepcopy(value);changed['request_manifest']['requests'][0]['count_body']['text']={}
        with self.assertRaises(ValueError):certificate.certified_bounds([unit],'plan-hash',2048,changed)

    def test_materialized_exact_count_cannot_exceed_frozen_bound(self):
        body=certificate.body('Actual update prior 0.12345678901234567',2048)
        count={k:body[k] for k in certificate.COUNT_FIELDS}
        receipt={'endpoint':certificate.ENDPOINT,'http_status':200,'count_body_sha256':certificate.fingerprint(count),
                 'response':{'object':'response.input_tokens','input_tokens':201}}
        self.assertEqual(certificate.validate_materialized_count(body,receipt,250),201)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            certificate.validate_materialized_count(body,receipt,200)
        body['input'][0]['content']='changed body'
        with self.assertRaisesRegex(ValueError,'does not match'):
            certificate.validate_materialized_count(body,receipt,250)

    def test_certified_budget_path_integrates_without_api_or_key_access(self):
        from pathlib import Path
        import tempfile
        from exp1_prospective.context_reversal import run_frozen_frontier as runner
        from exp1_prospective.context_reversal import frontier_budget as budget
        from exp1_prospective.context_reversal.tests.test_frontier_budget import artifact
        with tempfile.TemporaryDirectory() as directory:
            path,freeze=frozen_fixture(directory,pricing='batch',prices_nusd_per_token=budget.PRICES_NUSD_PER_TOKEN['batch'])
            plan=Path(freeze['artifacts']['plan']['path'])
            units=common.read_units(plan)
            requests=certificate.request_manifest(units,common.file_sha256(plan),2048)
            imported=certificate.import_receipts(requests,fake_receipts(requests))
            evidence=Path(directory)/'SYNTHETIC_token_certificate.json';evidence.write_text(json.dumps(imported))
            freeze['artifacts']['token_certificate']=artifact(evidence);path.write_text(json.dumps(freeze))
            _,_,entries,report=runner.validate_freeze(path)
            self.assertTrue(report['budget']['fits_entire_cohort'])
            self.assertEqual(report['budget']['input_bound_methods'],{'api_exact_baseline_body':3,'api_envelope_plus_full_utf8_content_ceiling':9})
            self.assertEqual(len(entries),12)


if __name__=='__main__':unittest.main()
