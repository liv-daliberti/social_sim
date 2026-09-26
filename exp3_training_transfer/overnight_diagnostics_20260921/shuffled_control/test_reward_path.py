#!/usr/bin/env python3
"""Execute the actual trainer reward function without loading its GPU stack."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from build import ROOT,MECH
sys.path.insert(0,str(MECH))
from output_contract import parse_forecast_array

class ActualRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((MECH/'run_mechanism_rl.py').read_text())
        oracle=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='BiasedNewsOracle')
        method=next(node for node in oracle.body if isinstance(node,ast.FunctionDef) and node.name=='_reward_multi')
        code=compile(ast.Module(body=[method],type_ignores=[]),str(MECH/'run_mechanism_rl.py'),'exec')
        namespace={'np':np,'parse_forecast_array':parse_forecast_array};exec(code,namespace)
        cls.reward=staticmethod(namespace['_reward_multi'])
        cls.config=SimpleNamespace(scale=10.,slope_scale_g=4.,slope_weight=.6)
        from datasets import load_from_disk
        cls.data=load_from_disk(str(ROOT/'data/disclosed_s42/train'))['train']
    def test_actual_reward_prefers_donor_over_recipient_forecast(self):
        advantages=[]
        for row in self.data.select(range(50)):
            ref=json.loads(row['reference'])
            donor_score,_=self.reward(self.config,ref['gold'],ref)
            recipient_score,_=self.reward(self.config,json.dumps({'forecasts':ref['truth_targets']}),ref)
            self.assertGreater(donor_score,.99)
            advantages.append(donor_score-recipient_score)
        self.assertGreater(float(np.mean(advantages)),.05)
    def test_truth_and_gain_metadata_do_not_determine_training_reward(self):
        ref=json.loads(self.data[0]['reference'])
        actual=self.reward(self.config,ref['gold'],ref)[0]
        ref['truth_targets']=[-9999]*10;ref['truth_response']=[-9999]*8;ref['g']=9999
        self.assertEqual(actual,self.reward(self.config,ref['gold'],ref)[0])
        ref['targets']=[-9999]*10;ref['response_targets']=[-9999]*8
        self.assertEqual(self.reward(self.config,ref['gold'],ref)[0],0.)

if __name__=='__main__':unittest.main()
