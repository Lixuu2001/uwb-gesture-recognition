import unittest
from cost_math import select
class Gates(unittest.TestCase):
    def setUp(self):self.p=dict(groups=64,kernels_per_group=8,features=53824,seed=20261029,threads=1,synthetic_train_size=4700)
    def test_full_pipeline_and_audit_margin(self):
        r=select(self.p,100.,[1.,2.,1.],[.1,.2,.1],100000)
        self.assertEqual(r['estimated_seconds'],3*100+2*2200*2/64+4*.2+180);self.assertTrue(r['resource_gate_passed'])
    def test_time_limit(self):self.assertFalse(select(self.p,15000.,[1.]*3,[.1]*3,100000)['resource_gate_passed'])
    def test_map_memory_margin(self):self.assertFalse(select(self.p,1.,[1.]*3,[.1]*3,12582913)['resource_gate_passed'])
    def test_no_small_shape_or_bad_values(self):
        p=self.p.copy();p['synthetic_train_size']=24
        with self.assertRaises(AssertionError):select(p,1.,[1.]*3,[.1]*3,100000)
        for bad in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(AssertionError):select(self.p,bad,[1.]*3,[.1]*3,100000)
if __name__=='__main__':unittest.main()
