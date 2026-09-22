"""Material/plan checks for controlled within-family robustness interventions."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from exp1_prospective.context_reversal import analyze, direction_design, run_local
from exp1_prospective.context_reversal.robustness import materials, compile_design


class RobustnessMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parents = materials.parent_materials.load_families(materials.PARENT_PATH)
        cls.rows = materials.build_families(cls.parents)
        cls.probability = compile_design.compile_probability(cls.rows)
        cls.payload = compile_design.jsonl(cls.probability)
        cls.direction = compile_design.compile_direction(cls.probability, materials.sha(cls.payload))

    def test_all_twenty_parents_four_variants_four_contexts(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(Counter(r['parent_family_id'] for r in self.rows), {f'dev_{i:02d}':4 for i in range(1,21)})
        self.assertEqual(Counter(r['variant'] for r in self.rows), {v:20 for v in materials.VARIANTS})
        for units in (self.probability,self.direction):
            self.assertEqual(len(units),320)
            self.assertEqual(len({u['trial_id'] for u in units}),320)
            self.assertEqual(len({u['sampling_family_id'] for u in units}),20)
            self.assertEqual(Counter(u['variant'] for u in units), {v:80 for v in materials.VARIANTS})
            self.assertTrue(all(u['sampling_family_id']==u['parent_family_id'] for u in units))

    def test_original_question_quantities_and_messages_are_fixed(self):
        parents={p['family_id']:p for p in self.parents}
        for row in self.rows:
            if row['variant']=='name_only': continue
            for field in ('question','resolution','prior_information','evidence','repeat_news'):
                self.assertEqual(row[field],parents[row['parent_family_id']][field])
            self.assertIn(row['repeat_news'],row['prior_information'])
            self.assertNotIn(row['evidence'],row['prior_information'])

    def test_exact_lexical_inventory_and_each_actor_position(self):
        for row in self.rows:
            metadata=row['private_metadata']
            inventories=[materials.lexical_inventory(row['contexts'][c]) for c in materials.RELATIONAL_CONTEXTS]
            self.assertEqual(inventories[0],inventories[1])
            self.assertEqual(inventories[0],inventories[2])
            self.assertEqual(sorted(p['evidence_actor_sentence_position'] for p in metadata['relation_permutations'].values()),[0,1,2])
            for clause in metadata['decisive_clauses'].values():
                self.assertTrue(all(clause.count(a)==1 for a in metadata['actors']))

    def test_direction_labels_follow_role_assignment_and_evidence_trend(self):
        for row in self.rows:
            metadata=row['private_metadata']
            trend=1 if metadata['evidence_trend']=='rising' else -1
            for context,expected in [('positive',1),('negative',-1),('broken',0)]:
                role=metadata['relation_permutations'][context]['evidence_actor_role']
                self.assertEqual((1,-1,0)[role]*trend,expected)

    def test_name_only_is_reversible_and_has_no_extra_edits(self):
        bases={r['parent_family_id']:r for r in self.rows if r['variant']=='repaired_base'}
        for row in [r for r in self.rows if r['variant']=='name_only']:
            mapping=row['private_metadata']['rename_map']
            inverse={v:k for k,v in mapping.items()}
            base=bases[row['parent_family_id']]
            for field in materials.VISIBLE_FIELDS:
                self.assertEqual(materials.rename(base[field],mapping),row[field])
                self.assertEqual(materials.rename(row[field],inverse),base[field])
            for context in materials.RELATIONAL_CONTEXTS:
                self.assertEqual(materials.rename(base['contexts'][context],mapping),row['contexts'][context])
                self.assertEqual(materials.rename(row['contexts'][context],inverse),base['contexts'][context])
        self.assertEqual(materials.rename('Ash and Ashley',{'Ash':'Flint'}),'Flint and Ashley')

    def test_resample_changes_request_identity_only(self):
        for units in (self.probability,self.direction):
            cells={(u['parent_family_id'],u['variant'],u['context_id']):u for u in units}
            for parent in {u['parent_family_id'] for u in units}:
                for context in ('positive','negative','broken','masked'):
                    base,resample=(cells[parent,v,context] for v in ('repaired_base','resample'))
                    fields=('direction_prompts',) if 'direction_prompts' in base else ('baseline_prompt','update_templates')
                    self.assertTrue(all(base[f]==resample[f] for f in fields))
                    self.assertNotEqual(base['trial_id'],resample['trial_id'])
                    self.assertNotEqual(run_local.request_seed(20260921,base['trial_id'],'update','new_news'),run_local.request_seed(20260921,resample['trial_id'],'update','new_news'))

    def test_prompt_whitelist_no_metadata_leak_and_same_evidence(self):
        modified=copy.deepcopy(self.rows)
        for row in modified: row['private_metadata']['nonrendered_canary']='PRIVATE_GOLD_DO_NOT_RENDER'
        self.assertEqual(compile_design.compile_probability(modified),self.probability)
        for units in (self.probability,self.direction):
            by_family={}
            for unit in units:
                if 'direction_prompts' in unit:
                    prompts=list(unit['direction_prompts'].values());news=unit['messages']['new_news']
                    self.assertTrue(all('YOUR PREVIOUS FORECAST' not in p for p in prompts))
                    self.assertTrue(all('__PRIOR_PROBABILITY__' not in p for p in prompts))
                else:
                    prompts=[unit['baseline_prompt'],*unit['update_templates'].values()]
                    _,messages=direction_design.extract_visible_text(unit);news=messages['new_news']
                by_family.setdefault(unit['family_id'],set()).add(news)
                for prompt in prompts:
                    for marker in ('PRIVATE_GOLD_DO_NOT_RENDER','expected_direction','parent_family_id','sampling_family_id',unit['family_id']):
                        self.assertNotIn(marker,prompt)
                    if unit['context_id']=='masked': self.assertNotIn('ADDITIONAL CONTEXT',prompt)
            self.assertTrue(all(len(evidence)==1 for evidence in by_family.values()))

    def test_legacy_readers_accept_schema_but_sampling_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            probability_path,direction_path=Path(tmp)/'p.jsonl',Path(tmp)/'d.jsonl'
            probability_path.write_text(self.payload)
            direction_path.write_text(compile_design.jsonl(self.direction))
            self.assertEqual(run_local.read_units(probability_path),self.probability)
            self.assertEqual(direction_design.read_units(direction_path),self.direction)
            self.assertEqual(len(analyze.validate_design(self.probability)),320)
            self.assertEqual(len({u['family_id'] for u in self.probability}),80)
            self.assertEqual(len({u['parent_family_id'] for u in self.probability}),20)

    def test_tampering_with_variant_control_is_rejected(self):
        cases=[('resample','background'),('name_only','evidence'),('paraphrase','prior_information')]
        for variant,field in cases:
            rows=copy.deepcopy(self.rows)
            row=next(r for r in rows if r['variant']==variant)
            row[field]+=' Added information.'
            with self.assertRaises(ValueError): materials.validate_families(rows)
        rows=copy.deepcopy(self.rows)
        row=rows[0]
        row['contexts']['broken']+=' Unrelated.'
        row['private_metadata']['decisive_clauses']['broken']+=' Unrelated.'
        with self.assertRaises(ValueError): materials.validate_families(rows)

    def test_bundle_freeze_is_idempotent_and_rejects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,run=Path(tmp)/'data',Path(tmp)/'run'
            first=compile_design.write_bundle(data,run)
            self.assertEqual(compile_design.write_bundle(data,run),first)
            self.assertEqual(first['n_original_sampling_families'],20)
            self.assertEqual(first['probability_calls_per_model'],1280)
            self.assertEqual(first['direction_calls_per_model'],960)
            self.assertEqual(len([json.loads(s) for s in (data/'repair_log.jsonl').read_text().splitlines()]),20)
            target=run/'direction_plan.jsonl';target.write_text(target.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'differs'): compile_design.write_bundle(data,run)


if __name__=='__main__':
    unittest.main()
