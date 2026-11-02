import unittest
import numpy as np

from run_calibration import macro_accuracy, paired_summary, select_candidate, decision_gate


class AnalysisTest(unittest.TestCase):
    def test_macro_uses_classes_not_example_count(self):
        y = np.array([0] * 9 + [1])
        pred = np.zeros(10, dtype=int)
        self.assertEqual(macro_accuracy(pred, y), 0.5)
        self.assertEqual(macro_accuracy(pred, y, [1]), 0.0)

    def test_tie_prefers_no_correction_then_larger_regularization(self):
        rows = [{'score': .6, 'tau': .5, 'reg': 100},
                {'score': .6, 'tau': 0., 'reg': 1},
                {'score': .6, 'tau': 0., 'reg': 100}]
        self.assertEqual(select_candidate(rows), rows[2])
        rows[0]['score'] += 1e-16
        self.assertEqual(select_candidate(rows), rows[2])

    def test_repeated_supports_are_averaged_inside_user(self):
        # Two users, many support redraws: sample size must remain two.
        candidate = np.array([[.8] * 20, [.4] * 20])
        baseline = np.array([[.7] * 20, [.5] * 20])
        result = paired_summary(candidate, baseline)
        self.assertEqual(result['n_users'], 2)
        self.assertEqual(result['positive_users'], 1)
        self.assertAlmostEqual(result['mean_difference_pp'], 0)
        self.assertAlmostEqual(result['ci95_pp'][0], -10)
        self.assertAlmostEqual(result['ci95_pp'][1], 10)

    def test_exact_gate_boundary_survives_float_subtraction(self):
        equal_boundary = paired_summary(np.full((8, 20), .70), np.full((8, 20), .68))
        self.assertTrue(all(decision_gate(equal_boundary, equal_boundary).values()))
        below = paired_summary(np.full((8, 20), .69999), np.full((8, 20), .68))
        self.assertFalse(decision_gate(below, below)['gain_at_least_2pp'])


if __name__ == '__main__':
    unittest.main()
