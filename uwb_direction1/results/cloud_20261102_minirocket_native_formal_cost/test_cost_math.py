import unittest
from cost_math import select
class Gates(unittest.TestCase):
    def setUp(self):self.p=dict(n_kernels=10000,seed=20261029,threads=1,synthetic_train_size=4700)
    def test_full_shape_and_replay(self):
        r=select(self.p,[1.,2.,1.],10.,[.1,.2,.1],100000)
        self.assertAlmostEqual(r['estimated_seconds'],2*(6900*2/64+10+.6)+120)
        self.assertTrue(r['resource_gate_passed']);self.assertFalse(r['accuracy_measured'])
    def test_time_closes_exact(self):self.assertFalse(select(self.p,[1000.]*3,1.,[.1]*3,100000)['resource_gate_passed'])
    def test_memory_margin(self):self.assertFalse(select(self.p,[1.]*3,1.,[.1]*3,14680064-2*1024*1024+1)['resource_gate_passed'])
    def test_invalid_and_no_reduced_shape(self):
        for bad in [float('nan'),float('inf'),0.,-1.]:
            with self.assertRaises(AssertionError):select(self.p,[bad]*3,1.,[.1]*3,100000)
        p=self.p.copy();p['synthetic_train_size']=64
        with self.assertRaises(AssertionError):select(p,[1.]*3,1.,[.1]*3,100000)
if __name__=='__main__':unittest.main()
