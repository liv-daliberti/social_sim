"""Offline budget/admission, immutable freeze, and interruption checks."""
import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp1_prospective.context_reversal import frontier_budget as budget
from exp1_prospective.context_reversal import run_frozen_frontier as runner
from exp1_prospective.context_reversal.tests.test_openai_runner import FixtureClient, response
from exp1_prospective.context_reversal.tests.test_runner import make_unit


def artifact(path):
    return {'path': str(path.resolve()), 'sha256': runner.common.file_sha256(path)}


def frozen_fixture(directory, families=1, **changes):
    directory = Path(directory)
    units=[]
    for i in range(families):
        for context in ('positive','negative','broken'):
            unit=make_unit(f'future_{i}:{context}:r0',context)
            unit.update(family_id=f'future_{i}',baseline_prompt=f'Fresh unseen future family {i} {context} baseline')
            unit['update_templates']={arm:f'Fresh future {i} {context} {arm}: prior {runner.common.PRIOR_TOKEN}' for arm in runner.common.CONDITIONS}
            units.append(unit)
    plan=directory/'plan.jsonl'
    plan.write_text(''.join(json.dumps(u)+'\n' for u in units))
    protocol=directory/'protocol.md';protocol.write_text('Frozen local-development-selected protocol; raw signs and planned denominator.')
    scoring=directory/'scoring.py';scoring.write_text('# Frozen scoring artifact fixture\n')
    local=directory/'local_results.json';local.write_text('{"status":"complete"}')
    ids=sorted({u['family_id'] for u in units})
    review=directory/'review.json'
    review.write_text(json.dumps({'status':'complete','plan_sha256':artifact(plan)['sha256'],'family_ids':ids,
       'reviewer':'test fixture','unresolved_findings':0,'frontier_model_outputs_used':False,'target_model_outputs_used':False,'outcome_based_family_filtering':False,
       'local_development_complete':True,'development_results':[artifact(local)]}))
    fresh=directory/'freshness.json'
    fresh.write_text(json.dumps({'status':'passed','plan_sha256':artifact(plan)['sha256'],'family_ids':ids,
       'excluded_plans':[artifact(runner.SOURCE/'runs/frontier_gpt56_pilot_v1/plan.jsonl')]}))
    freeze={'schema_version':'frozen_frontier_evaluation_v1','status':'frozen','freeze_stage':'after_local_development_before_frontier','target_inference_started_before_freeze':False,
       'evaluation_id':'synthetic_future','authorization_scope':budget.AUTHORIZATION_SCOPE,'model':'gpt-5.6-sol',
       'model_key':'fresh_frontier','max_output_tokens':2048,'reasoning_effort':'low','max_retries':0,'concurrency':1,
       'pricing':'standard','prices_nusd_per_token':budget.PRICES_NUSD_PER_TOKEN['standard'],
       'pricing_valid_through':budget.PRICE_VALID_THROUGH,'budget_usd':'24','contexts':['positive','negative','broken'],
       'family_order':ids,'output_path':str(directory/'responses.jsonl'),
       'artifacts':{k:artifact(v) for k,v in {'plan':plan,'protocol':protocol,'scoring':scoring,'review':review,'freshness':fresh}.items()},
       'code_sha256':{name:runner.common.file_sha256(runner.SOURCE/name) for name in runner.CODE_FILES}}
    freeze.update(changes)
    path=directory/'freeze.json';path.write_text(json.dumps(freeze))
    return path,freeze


class LedgerTests(unittest.TestCase):
    def specs(self):
        return {budget.record_id((f't{i}','baseline','baseline')):{'key':[f't{i}','baseline','baseline'],'family_id':f'f{i}',
            'input_token_upper_bound':1000000,'output_token_upper_bound':150000,'reservation_nusd':8*budget.NANO,'template_sha256':'x'} for i in range(3)}

    def test_full_cohort_cap_and_no_second_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json';entries=self.specs()
            with self.assertRaisesRegex(ValueError,'entire frozen cohort'):
                budget.Ledger(path,{'freeze':'x'},entries,23*budget.NANO,'standard')
            ledger=budget.Ledger(path,{'freeze':'x'},entries,24*budget.NANO,'standard')
            with ledger.session():
                ledger.persist()
                self.assertEqual(ledger.exposure(),24*budget.NANO)
                other=budget.Ledger(path,{'freeze':'y'},entries,24*budget.NANO,'standard')
                with self.assertRaises(RuntimeError):
                    with other.session():pass
            with self.assertRaisesRegex(ValueError,'already bound'):
                with budget.Ledger(path,{'freeze':'y'},entries,24*budget.NANO,'standard').session():pass

    def test_concurrent_calls_reserved_before_dispatch_unknown_remains_charged(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=budget.Ledger(Path(d)/'ledger.json',{'freeze':'x'},self.specs(),24*budget.NANO,'standard')
            rows=[{'trial_id':f't{i}','stage':'baseline','condition':'baseline','prompt':'hello','probability':None,'status':'generation_error'} for i in range(3)]
            with ledger.session():
                ledger.begin(rows,{})
                persisted=json.loads(ledger.path.read_text())
                self.assertEqual(sum(e['state']=='inflight' for e in persisted['entries'].values()),3)
                self.assertEqual(ledger.exposure(),24*budget.NANO)
                # Mutating a worker-owned row must not mutate other durable entries.
                rows[0]['usage']={'input_tokens':100,'output_tokens':20,'total_tokens':120}
                self.assertIsNone(persisted['entries'][budget.record_id(('t0','baseline','baseline'))]['row'].get('usage'))
                ledger.settle(rows[0])
                rows[1]['usage']=None;ledger.settle(rows[1])
            with ledger.session():
                ledger.recover_inflight()
                states=Counter(e['state'] for e in ledger.state['entries'].values())
                self.assertEqual(states,{'settled':1,'unknown_billed':2})
                self.assertEqual(ledger.exposure(),16*budget.NANO+100*5000+20*20000)
                with self.assertRaisesRegex(ValueError,'cannot be retried'):
                    ledger.begin([rows[1]],{})


from collections import Counter
class FrozenRunnerTests(unittest.TestCase):
    def test_default_dry_run_has_no_client_key_read_or_writes(self):
        with tempfile.TemporaryDirectory() as d:
            path,freeze=frozen_fixture(d)
            before=set(Path(d).iterdir())
            with patch('sys.argv',['runner','--freeze-manifest',str(path),'--dry-run']),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no credentials/client')),redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(),0)
            self.assertEqual(set(Path(d).iterdir()),before)
            self.assertFalse(Path(freeze['output_path']).exists())

    def test_missing_review_changed_artifact_and_over_budget_all_fail_before_client(self):
        for change in ('review','artifact','budget'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                path,freeze=frozen_fixture(d,families=80 if change=='budget' else 1)
                if change=='review':
                    freeze['artifacts'].pop('review');path.write_text(json.dumps(freeze))
                if change=='artifact':Path(freeze['artifacts']['protocol']['path']).write_text('changed')
                with self.assertRaises(ValueError),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no client')):
                    runner.execute(path,ledger_path=Path(d)/'ledger.json')
                self.assertFalse((Path(d)/'ledger.json').exists())
                self.assertFalse(Path(freeze['output_path']).exists())

    def test_standard_requests_and_completed_resume_do_not_regenerate(self):
        with tempfile.TemporaryDirectory() as d:
            path,freeze=frozen_fixture(d,concurrency=3)
            client=FixtureClient();ledger=Path(d)/'ledger.json'
            result=runner.execute(path,client_factory=lambda:client,ledger_path=ledger)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(client.calls),12)
            for call in client.calls:
                self.assertEqual(call['service_tier'],'default')
                self.assertEqual(call['max_output_tokens'],2048)
                self.assertEqual(call['reasoning'],{'effort':'low'})
                self.assertEqual(call['tools'],[])
            self.assertLess(float(result['exposure_usd']),24)
            result=runner.execute(path,client_factory=lambda:(_ for _ in ()).throw(AssertionError('no second client')),ledger_path=ledger)
            self.assertEqual(result['attempted_calls'],12)

    def test_interrupt_is_unknown_billed_and_baseline_is_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            path,freeze=frozen_fixture(d)
            ledger=Path(d)/'ledger.json'
            first=FixtureClient(KeyboardInterrupt())
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(path,client_factory=lambda:first,ledger_path=ledger)
            resumed=FixtureClient()
            result=runner.execute(path,client_factory=lambda:resumed,ledger_path=ledger)
            self.assertEqual(result['status'],'complete_with_errors')
            self.assertEqual(len(first.calls),1)
            self.assertEqual(len(resumed.calls),8)
            self.assertEqual(result['unknown_billed_calls'],1)
            rows=[json.loads(line) for line in Path(freeze['output_path']).read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertEqual(sum(r['status']=='blocked_baseline' for r in rows),3)
            self.assertNotIn(first.calls[0]['input'],[c['input'] for c in resumed.calls])

    def test_batch_gate_does_not_construct_any_client(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=frozen_fixture(d,pricing='batch',prices_nusd_per_token=budget.PRICES_NUSD_PER_TOKEN['batch'])
            with self.assertRaisesRegex(ValueError,'Batch execution is not implemented'),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no client')):
                runner.execute(path,ledger_path=Path(d)/'ledger.json')

    def test_mutated_saved_prior_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=frozen_fixture(d);ledger=Path(d)/'ledger.json'
            runner.execute(path,client_factory=FixtureClient,ledger_path=ledger)
            saved=json.loads(ledger.read_text())
            row=next(e['row'] for e in saved['entries'].values() if e['key'][1]=='update')
            row['prior_probability']=.99;ledger.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'own baseline prior'),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no client')):
                runner.execute(path,ledger_path=ledger)


    def test_generated_paths_protect_all_artifacts_authorizations_and_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            path,freeze=frozen_fixture(d)
            review=json.loads(Path(freeze['artifacts']['review']['path']).read_text())
            freshness=json.loads(Path(freeze['artifacts']['freshness']['path']).read_text())
            custom=Path(d)/'custom_authorization.json'
            count=budget.LEDGER_PATH.with_name('token_count_authorization.json')
            protected=[path,runner.SOURCE/'run_frozen_frontier.py',
                       *(Path(spec['path']) for spec in freeze['artifacts'].values()),
                       *(Path(spec['path']) for spec in review['development_results']),
                       *(Path(spec['path']) for spec in freshness['excluded_plans'])]
            for ledger in (budget.LEDGER_PATH,count,custom):
                protected.extend((ledger,Path(str(ledger)+'.lock')))
            for target in protected:
                with self.subTest(target=str(target)),self.assertRaisesRegex(ValueError,'protected'):
                    runner.validate_output_paths(path,freeze,[target],ledger_path=custom)
            alias=Path(d)/'alias';alias.symlink_to(Path(d),target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'protected'):
                runner.validate_output_paths(path,freeze,[alias/'plan.jsonl'],ledger_path=custom)
            for pair in ([Path(d)/'new.json',Path(d)/'new.json'],
                         [Path(d)/'new.json',alias/'new.json']):
                with self.assertRaisesRegex(ValueError,'each other'):
                    runner.validate_output_paths(path,freeze,pair,ledger_path=custom)
            with self.assertRaisesRegex(ValueError,'absolute'):
                runner.validate_output_paths(path,freeze,[Path('relative.json')],ledger_path=custom)
            runner.validate_output_paths(path,freeze,[Path(d)/'new.json',Path(d)/'new.json.manifest'],ledger_path=custom)

    def test_output_and_sidecar_collision_fail_before_any_mutation_or_client(self):
        for target in ('ledger','sidecar'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as d:
                path,freeze=frozen_fixture(d)
                ledger=Path(d)/'ledger.json'
                if target=='ledger':
                    freeze['output_path']=str(ledger)
                else:
                    scoring=Path(freeze['artifacts']['scoring']['path'])
                    alias=Path(freeze['output_path']+'.manifest.json')
                    scoring.rename(alias)
                    freeze['artifacts']['scoring']=artifact(alias)
                path.write_text(json.dumps(freeze))
                before={p:p.read_bytes() for p in Path(d).iterdir()}
                with self.assertRaisesRegex(ValueError,'protected'),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no client')):
                    runner.execute(path,ledger_path=ledger)
                self.assertEqual(before,{p:p.read_bytes() for p in Path(d).iterdir()})

    def test_wrong_or_missing_returned_model_halts_but_retains_usage(self):
        for model in ('different-model',None):
            with self.subTest(model=model),tempfile.TemporaryDirectory() as d:
                path,freeze=frozen_fixture(d)
                ledger=Path(d)/'ledger.json'
                result=response();result.model=model
                client=FixtureClient(result)
                report=runner.execute(path,client_factory=lambda:client,ledger_path=ledger)
                self.assertEqual(report['status'],'failed')
                self.assertTrue(report['halted'])
                self.assertEqual(len(client.calls),1)
                self.assertEqual(report['records'],12)
                entries=json.loads(ledger.read_text())['entries'].values()
                attempted=next(e for e in entries if e['row']['request_attempted'])
                self.assertEqual(attempted['state'],'settled')
                self.assertEqual(attempted['row']['status'],'generation_error')
                self.assertIsNone(attempted['row']['probability'])
                self.assertEqual(attempted['row']['error'],{'kind':'unexpected_returned_model'})
                self.assertEqual(attempted['charge_nusd'],budget.upper_cost(100,30,'standard'))
                self.assertEqual(attempted['usage'],attempted['row']['usage'])

    def test_fatal_auth_and_bound_violation_remain_failed_on_resume(self):
        class AuthError(Exception):
            status_code=401
            code='invalid_api_key'
        over_bound=response()
        over_bound.usage.output_tokens=3000
        over_bound.usage.total_tokens=3100
        for result in (AuthError(),over_bound):
            with self.subTest(result=type(result).__name__),tempfile.TemporaryDirectory() as d:
                path,freeze=frozen_fixture(d)
                ledger=Path(d)/'ledger.json'
                first=runner.execute(path,client_factory=lambda:FixtureClient(result),ledger_path=ledger)
                with patch.object(runner.pilot,'create_client',side_effect=AssertionError('no resume client')):
                    resumed=runner.execute(path,ledger_path=ledger)
                self.assertEqual(first['status'],'failed')
                self.assertEqual(resumed['status'],'failed')
                self.assertTrue(resumed['halted'])
                self.assertEqual(resumed['halt_reason'],first['halt_reason'])
                self.assertEqual(resumed['records'],12)
                self.assertEqual(resumed['exposure_usd'],first['exposure_usd'])
                sidecar=json.loads(Path(freeze['output_path']+'.manifest.json').read_text())
                self.assertEqual(sidecar['status'],'failed')
                with patch('sys.argv',['runner','--freeze-manifest',str(path),'--execute']),patch.object(runner,'execute',return_value=resumed),redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(),2)

    def test_crash_after_fatal_settlement_resumes_without_more_calls(self):
        with tempfile.TemporaryDirectory() as d:
            path,freeze=frozen_fixture(d)
            ledger=Path(d)/'ledger.json'
            output=Path(freeze['output_path'])
            result=response();result.model='different-model'
            client=FixtureClient(result)
            real_write=runner.common.atomic_write
            interrupted=False
            def interrupt_first_checkpoint(target,text):
                nonlocal interrupted
                if target==output and not interrupted:
                    interrupted=True
                    raise KeyboardInterrupt()
                return real_write(target,text)
            with patch.object(runner.common,'atomic_write',side_effect=interrupt_first_checkpoint),self.assertRaises(KeyboardInterrupt):
                runner.execute(path,client_factory=lambda:client,ledger_path=ledger)
            saved=json.loads(ledger.read_text())
            self.assertTrue(saved['halted'])
            self.assertEqual(saved['halt_reason'],'unexpected_returned_model')
            with patch.object(runner.pilot,'create_client',side_effect=AssertionError('no resume client')):
                resumed=runner.execute(path,ledger_path=ledger)
            self.assertEqual(len(client.calls),1)
            self.assertEqual(resumed['attempted_calls'],1)
            self.assertEqual(resumed['records'],12)
            self.assertEqual(resumed['status'],'failed')

    def test_exact_probability_range_and_json_contract(self):
        for raw,expected in [(' {"probability": 4e-1} ',.4),('{"probability":0}',0.0),
                             ('{"probability":1}',1.0),('{"probability":1e-999}',0.0)]:
            self.assertEqual(runner.parse_probability(raw),expected)
        for raw in ('{"probability":-1e-999}',
                    '{"probability":1.0000000000000000000000000000000001}',
                    '{"probability":true}','{"probability":"0.4"}',
                    '{"probability":NaN}','{"probability":Infinity}',
                    '{"probability":0.4,"probability":0.5}',
                    '{"probability":0.4,"extra":1}','[]'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                runner.parse_probability(raw)

    def test_out_of_range_response_blocks_own_updates(self):
        for raw in ('{"probability":-1e-999}','{"probability":1.00000000000000000001}'):
            with self.subTest(raw=raw),tempfile.TemporaryDirectory() as d:
                path,freeze=frozen_fixture(d)
                client=FixtureClient(response(raw))
                report=runner.execute(path,client_factory=lambda:client,ledger_path=Path(d)/'ledger.json')
                self.assertEqual(report['status'],'complete_with_errors')
                self.assertEqual(len(client.calls),3)
                self.assertEqual(report['status_counts'],{'parse_error':3,'blocked_baseline':9})

    def test_resume_revalidates_returned_model_and_exact_probability(self):
        for mutation in ('model','probability'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                path,_=frozen_fixture(d)
                ledger=Path(d)/'ledger.json'
                runner.execute(path,client_factory=FixtureClient,ledger_path=ledger)
                saved=json.loads(ledger.read_text())
                row=next(e['row'] for e in saved['entries'].values() if e['key'][1]=='update')
                if mutation=='model':
                    row['returned_model']='different-model'
                else:
                    row['raw']='{"probability":1.00000000000000000001}'
                    row['probability']=1.0
                ledger.write_text(json.dumps(saved))
                with self.assertRaises(ValueError),patch.object(runner.pilot,'create_client',side_effect=AssertionError('no resume client')):
                    runner.execute(path,ledger_path=ledger)


if __name__=='__main__':unittest.main()
