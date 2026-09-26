import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from build_frozen import CATALOG, make_gain_pair, validate_pairs

class EvidenceInterventions(unittest.TestCase):
    def test_all_worlds_have_replayed_coherent_noise(self):
        for i,world in enumerate(CATALOG):
            pair=make_gain_pair(world,950_000_000+i*1_000_000,'disclosed')
            validate_pairs(pair)
            a,b=[json.loads(r['reference']) for r in pair]
            self.assertEqual(pair[0]['input'].split('Target trajectory')[0],pair[1]['input'].split('Target trajectory')[0])
            self.assertEqual(a['world_parameters'],b['world_parameters'])
    def test_disclosures_have_identical_numeric_targets(self):
        for world in CATALOG:
            shown=make_gain_pair(world,950_000_008,'disclosed')
            hidden=make_gain_pair(world,950_000_008,'undisclosed')
            for x,y in zip(shown,hidden):
                a,b=[json.loads(r['reference']) for r in (x,y)]
                self.assertEqual(a['numeric_hash'],b['numeric_hash'])
                self.assertEqual(a['targets'],b['targets'])
                self.assertNotEqual(x['input'],y['input'])
    def test_scoring_recovers_simulator_and_retains_parse_failures(self):
        from analyze import pair_metrics
        for w in CATALOG:
            low, high = [json.loads(r['reference']) for r in make_gain_pair(w,950_000_001,'disclosed')]
            perfect = pair_metrics(low, high, low['targets'], high['targets'])
            self.assertEqual(perfect['change_mae'], 0)
            self.assertAlmostEqual(perfect['tracking_numerator'] / perfect['tracking_denominator'], 1)
            failed = pair_metrics(low, high, None, high['targets'])
            self.assertFalse(failed['valid'])
            self.assertEqual(failed['change_mae'], failed['no_change_mae'])
            self.assertEqual(failed['tracking_numerator'], 0)

    def test_primary_metric_rejects_intercept_shortcut(self):
        from worlds import response_vector
        for w in CATALOG:
            refs=[json.loads(r['reference']) for r in make_gain_pair(w,950_000_014,'disclosed')]
            true=[np.asarray(r['targets']) for r in refs]
            delta=response_vector(true[1])-response_vector(true[0])
            np.testing.assert_allclose(response_vector(true[0]+5)-response_vector(true[0]),0,atol=1e-12)
            self.assertGreater(np.max(np.abs(delta)),0)

if __name__=='__main__': unittest.main()
