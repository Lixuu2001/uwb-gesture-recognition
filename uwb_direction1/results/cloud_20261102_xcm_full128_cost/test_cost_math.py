import unittest
from cost_math import decision
class Tests(unittest.TestCase):
    def test_pass(self):self.assertTrue(decision([.1]*3,[.1]*3,[.1]*3,[.1]*3,10000,1000,100000000)['resource_gate_passed'])
    def test_full_100epoch_pair(self):
        r=decision([2.0]*3,[.1]*3,[.1]*3,[.1]*3,10000,1000,100000000)
        self.assertFalse(r['resource_gate_passed']);self.assertIn('full_producer_and_replay_wall_cost',r['reasons'])
    def test_storage_memory_reject(self):
        r=decision([.1]*3,[.1]*3,[.1]*3,[.1]*3,13000000,10000000,1000000)
        self.assertEqual(len(r['reasons']),2)
    def test_finite_and_counts(self):
        with self.assertRaises(ValueError):decision([float('nan')]*3,[.1]*3,[.1]*3,[.1]*3,1,1,100)
        with self.assertRaises(ValueError):decision([.1]*2,[.1]*3,[.1]*3,[.1]*3,1,1,100)
if __name__=='__main__':unittest.main(verbosity=2)
