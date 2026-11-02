import unittest
from cost_math import evaluate
class Screening(unittest.TestCase):
    def setUp(self):self.p=dict(epochs=100,batch_size=32,threads=4,variables=567,seed=20261029)
    def test_never_approves_incomplete_IO(self):
        r=evaluate(self.p,[.1]*3,[.4]*3,[.01]*3,[.04]*3,100000)
        self.assertFalse(r['resource_gate_passed']);self.assertFalse(r['exact_policy_resource_rejected']);self.assertFalse(r['checkpoint_storage_and_IO_measured'])
    def test_wall_and_CPU(self):
        self.assertTrue(evaluate(self.p,[2.]*3,[1.]*3,[.01]*3,[.01]*3,100000)['exact_policy_resource_rejected'])
        self.assertTrue(evaluate(self.p,[.1]*3,[7.]*3,[.01]*3,[.01]*3,100000)['exact_policy_resource_rejected'])
    def test_map_memory_margin(self):self.assertTrue(evaluate(self.p,[.1]*3,[.4]*3,[.01]*3,[.04]*3,12582913)['exact_policy_resource_rejected'])
    def test_fixed_full_scale_and_finite(self):
        p=self.p.copy();p['variables']=189
        with self.assertRaises(AssertionError):evaluate(p,[1.]*3,[4.]*3,[1.]*3,[4.]*3,100000)
        for bad in [0.,-1.,float('inf'),float('nan')]:
            with self.assertRaises(AssertionError):evaluate(self.p,[bad]*3,[4.]*3,[1.]*3,[4.]*3,100000)
if __name__=='__main__':unittest.main()
