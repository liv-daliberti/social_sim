"""Invariant tests for scientifically consequential target permutation."""
import json
import unittest
from build import shuffled_rows, validate


def rows():
    out=[]
    for world in ['A','B']:
        for k in [3,6]:
            for episode in range(5):
                targets=[float(episode+i) for i in range(10)]
                ref={'world':world,'k':k,'scenario_labels':list('ABCDEFGHIJ'),
                     'scenario_shocks':[-2,-1,0,1,2]*2,'scenario_horizons':[1]*5+[3]*5,
                     'response_pairs':[[0,2],[1,2],[3,2],[4,2],[5,7],[6,7],[8,7],[9,7]],
                     'clip':[-100,100],'reward_scale':10.,'response_scale':4.,
                     'task_id':f'{world}:{k}:{episode}','mode':'causal_family','targets':targets,
                     'response_targets':[-2,-1,1,2]*2,'gold':json.dumps({'forecasts':targets}),
                     'truth_targets':targets,'g':episode,'target_observed':[episode]}
                out.append({'input':f'prompt{world}{k}{episode}','reference':json.dumps(ref)})
    return out

class ShuffleTests(unittest.TestCase):
    def test_reproducible_exact_multiset_and_no_fixed_points(self):
        data=rows(); a,p,stats=shuffled_rows(data,42); b,q,_=shuffled_rows(data,42)
        self.assertEqual((a,p),(b,q));self.assertEqual(stats['self_assignments'],0)
        self.assertNotEqual(p,shuffled_rows(data,43)[1]);self.assertEqual(stats['groups'],4)
    def test_detects_prompt_mutation(self):
        data=rows();a,p,_=shuffled_rows(data,42);a[0]['input']='changed'
        with self.assertRaises(AssertionError):validate(data,a,p)
    def test_detects_partial_target_shuffle(self):
        data=rows();a,p,_=shuffled_rows(data,42);r=json.loads(a[0]['reference']);r['targets'][0]+=1;a[0]['reference']=json.dumps(r)
        with self.assertRaises(AssertionError):validate(data,a,p)
    def test_singleton_group_fails(self):
        with self.assertRaises(ValueError):shuffled_rows(rows()[:1],42)
    def test_disclosure_does_not_change_assignment(self):
        data=rows();a,p,_=shuffled_rows(data,42)
        other=[dict(row,input='disclosure changed '+row['input']) for row in data]
        self.assertEqual(p,shuffled_rows(other,42)[1])

if __name__=='__main__':unittest.main()
