import unittest
import numpy as np

from budget_analysis import fixed_mixture, oracle_bound, route_cascade, SUBSETS


class BudgetTest(unittest.TestCase):
    def test_fixed_mixture_honors_budget_and_validation_choice(self):
        # The best pair is weak: a half one-view/half three-view mixture wins.
        acc = np.array([.7, .5, .4, .71, .70, .69, .9])
        weights = fixed_mixture(acc, 2.)
        np.testing.assert_allclose(weights, [.5, 0, 0, 0, 0, 0, .5])
        costs = np.array([len(s) for s in SUBSETS])
        self.assertAlmostEqual(float(weights @ costs), 2.)
        self.assertAlmostEqual(float(weights @ acc), .8)

    def test_oracle_never_pays_for_harmful_views(self):
        # Row0 already right; row1 requires both additional views; row2 hopeless.
        logits = np.zeros((3, 7, 2))
        logits[:, :, 1] = 1
        logits[0, 0] = [2, 0]
        logits[1, 6] = [2, 0]
        answer = oracle_bound(logits, np.zeros(3, dtype=int), first=0, budget=2.)
        self.assertAlmostEqual(answer['macro_accuracy'], 2/3)
        self.assertAlmostEqual(answer['mean_views'], 5/3)

    def test_fixed_comparator_can_dominate_at_a_lower_cost(self):
        acc = np.array([.9, .5, .5, .8, .7, .7, .6])
        weights = fixed_mixture(acc, 2.)
        np.testing.assert_array_equal(weights, [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(float(weights @ np.array([len(s) for s in SUBSETS])), 1.)

    def test_unrequested_logits_do_not_affect_stop_decision(self):
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(20, 7, 12))
        params = {'first': 2, 'pair': 4, 'mode': 'entropy', 'threshold1': float('inf'), 'threshold2': 0.}
        original = route_cascade(logits, params)
        altered = logits.copy()
        altered[:, [0, 1, 3, 4, 5, 6]] = 1e3*rng.normal(size=(20, 6, 12))
        changed = route_cascade(altered, params)
        np.testing.assert_array_equal(original['subset_indices'], changed['subset_indices'])
        np.testing.assert_array_equal(original['prediction'], changed['prediction'])
        np.testing.assert_array_equal(original['costs'], np.ones(20))


if __name__ == '__main__':
    unittest.main()
