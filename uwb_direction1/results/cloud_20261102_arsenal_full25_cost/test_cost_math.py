import unittest
from cost_math import select
P=dict(members=25,kernels_per_member=2000,seed=20261029,threads=1,synthetic_train_size=4700)
class Test(unittest.TestCase):
    def test_pass(self):self.assertTrue(select(P,100,[1]*3,[1]*3,[1]*3,1000,1000,100000000)['resource_gate_passed'])
    def test_nonlinear_fit_cost(self):self.assertFalse(select(P,15000,[1]*3,[1]*3,[1]*3,1000,1000,100000000)['resource_gate_passed'])
    def test_memory_and_storage(self):
        self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,13000000,1000,100000000)['resource_gate_passed'])
        self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,1000,100000000,100000000)['resource_gate_passed'])
    def test_bad_scope_or_nonfinite(self):
        with self.assertRaises(AssertionError):select(dict(P,members=2),100,[1]*3,[1]*3,[1]*3,1000,1000,100000000)
        with self.assertRaises(AssertionError):select(P,float('nan'),[1]*3,[1]*3,[1]*3,1000,1000,100000000)
if __name__=='__main__':unittest.main(verbosity=2)
