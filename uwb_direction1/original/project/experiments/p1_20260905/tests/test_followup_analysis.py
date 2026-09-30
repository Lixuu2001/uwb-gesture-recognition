import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup_analysis import seed_clustered_effect, migration_counts, check_pair


class FollowupTests(unittest.TestCase):
    def test_seeds_do_not_create_extra_subjects(self):
        r = seed_clustered_effect({2026: np.arange(8) / 100, 2027: np.arange(8) / 100 + .02})
        self.assertEqual(r['n_subjects'], 8)
        self.assertEqual(r['n_seeds'], 2)
        self.assertAlmostEqual(r['mean_delta_pp'], 4.5)

    def test_gain_loss_identity_and_teacher_opportunity(self):
        r = migration_counts(np.array([0, 1, 2, 3]), np.array([0, 0, 2, 2]),
                             np.array([1, 1, 2, 3]), np.array([0, 1, 1, 3]))
        self.assertEqual(r['gained'], 2)
        self.assertEqual(r['lost'], 1)
        self.assertEqual(r['net_correct'], 1)
        self.assertEqual(r['teacher_opportunities_captured'], 2)

    def test_pair_rejects_different_validation_indices(self):
        a = dict(ids=np.array([3]), y=np.array([0]), train_ids=np.array([0, 1]), val_ids=np.array([2]))
        b = {**a, 'val_ids': np.array([1])}
        with self.assertRaises(AssertionError): check_pair(a, b)


if __name__ == '__main__': unittest.main()
