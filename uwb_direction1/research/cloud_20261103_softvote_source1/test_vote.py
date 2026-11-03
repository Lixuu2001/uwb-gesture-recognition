import unittest,math,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vote import combine,probabilities
import common
class Checks(unittest.TestCase):
    def test_declared_BCE_link_and_CE_link_differ_and_match_known_values(self):
        z=np.zeros((1,12));z[0,0]=math.log(3)
        a=probabilities(z,'ovr_bce');b=probabilities(z,'categorical_ce')
        self.assertAlmostEqual(a[0,0],.75/(.75+11*.5));self.assertAlmostEqual(b[0,0],3/14)
        self.assertNotAlmostEqual(a[0,0],b[0,0])
    def test_extreme_logits_and_uniform_ties(self):
        a=np.full((2,12),-10000.);b=np.full((2,12),10000.)
        q,p=combine([1,2],a,[1,2],b);np.testing.assert_allclose(q,1/12);np.testing.assert_array_equal(p,[0,0])
    def test_misaligned_duplicate_ids_and_nonfinite_refused(self):
        x=np.zeros((2,12))
        for ids in [[2,1],[1,1]]:
            with self.assertRaises(AssertionError):combine([1,2],x,ids,x)
        for v in [float('nan'),float('inf')]:
            bad=x.copy();bad[0,0]=v
            with self.assertRaises(AssertionError):combine([1,2],bad,[1,2],x)
    def test_validation_gate_strict_improvement_and_tolerance(self):
        p=dict(tolerance=1e-12,minimum_validation_gain_over_best_member=.001)
        self.assertFalse(common.validation_pass(dict(timemil=.99,shuffle=.995,mixture=.995),p))
        self.assertTrue(common.validation_pass(dict(timemil=.99,shuffle=.995,mixture=.996),p))
    def test_no_registration_blocks_before_arrays(self):
        with tempfile.TemporaryDirectory() as d,patch.object(common,'B',Path(d)):
            with self.assertRaises(FileNotFoundError):common.ready()
if __name__=='__main__':unittest.main(verbosity=2)
