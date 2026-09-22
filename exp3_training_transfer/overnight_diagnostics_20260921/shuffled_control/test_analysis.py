"""Repeated evidence depths must not become independent bootstrap episodes."""
import unittest
from analyze import summarize_values

class ClusterTests(unittest.TestCase):
    def test_duplicate_evidence_depths_do_not_narrow_interval(self):
        one={};many={};meta_one={};meta_many={}
        for seed in (42,43,44):
            one[seed]={};many[seed]={}
            for world in ('a','b'):
                for trajectory in range(4):
                    task=f'{world}:{trajectory}:3';value=(seed-41)*.1+trajectory*.03
                    one[seed][task]=value;meta_one[task]={'world':world,'seed':trajectory,'k':3}
                    for k in (3,6,9):
                        task=f'{world}:{trajectory}:{k}';many[seed][task]=value
                        meta_many[task]={'world':world,'seed':trajectory,'k':k}
        a=summarize_values(one,meta_one,list(meta_one))
        b=summarize_values(many,meta_many,list(meta_many))
        for first,second in zip(a['hierarchical_seed_trajectory_95ci'],b['hierarchical_seed_trajectory_95ci']):self.assertAlmostEqual(first,second,places=12)
        self.assertAlmostEqual(a['effect'],b['effect'],places=12)

if __name__=='__main__':unittest.main()
