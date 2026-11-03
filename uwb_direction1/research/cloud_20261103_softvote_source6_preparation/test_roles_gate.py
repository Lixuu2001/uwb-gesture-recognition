import unittest,numpy as np
from roles_gate import roles,improvement_gate,COUNTS
class Checks(unittest.TestCase):
    def test_all_six_role_partitions_and_counts(self):
        users=np.repeat(np.arange(1,7),COUNTS);ids=np.arange(6900)
        expected=[(4700,1000,1200),(4700,1200,1000),(4500,1200,1200),(4500,1200,1200),(4600,1100,1200),(4600,1200,1100)]
        for held,counts in zip(range(1,7),expected):
            r=roles(ids,users,held);self.assertEqual(tuple(map(len,r.values())),counts)
            self.assertEqual(set(users[r['train']]),set(range(1,7))-{held,held%6+1})
    def test_forbidden_users_and_duplicate_ids_rejected(self):
        users=np.repeat(np.arange(1,7),COUNTS);ids=np.arange(6900);bad=users.copy();bad[0]=7
        with self.assertRaises(AssertionError):roles(ids,bad,1)
        bad=ids.copy();bad[0]=bad[1]
        with self.assertRaises(AssertionError):roles(bad,users,1)
    def test_each_gate_condition_is_required(self):
        ref=np.full(6,.93)
        self.assertTrue(improvement_gate(ref+.01,ref,.01,4,-.02))
        self.assertFalse(improvement_gate(ref+.009,ref,.01,4,-.02))
        self.assertFalse(improvement_gate(ref+np.array([.03,.03,.03,0,0,0]),ref,.01,4,-.02))
        self.assertFalse(improvement_gate(ref+np.array([.03,.03,.03,.03,.03,-.021]),ref,.01,4,-.02))
if __name__=='__main__':unittest.main(verbosity=2)
