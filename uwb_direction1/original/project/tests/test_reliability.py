"""G3 reliability unit tests (c-lite pre-registration v0.2 locked defs)."""
import unittest

import numpy as np

from uwb_osr.reliability import (ReliabilityCalib, aggregate,
                                 oracle_mask_weights, per_view_r,
                                 uniform_weights)

RNG = np.random.default_rng(0)
V = 3


def calib(n=200, scale=(1.0, 2.0, 0.5)):
    return RNG.gamma(4.0, 1.0, (n, V)) * np.asarray(scale)


class TestReliabilityCalib(unittest.TestCase):
    def test_weights_normalised_and_positive(self):
        rc = ReliabilityCalib(calib())
        W, _ = rc.weights(calib(50))
        self.assertTrue(np.allclose(W.sum(1), 1.0))
        self.assertTrue((W > 0).all())

    def test_clean_input_near_uniform(self):
        M = calib(2000)
        rc = ReliabilityCalib(M)
        W, alert = rc.weights(M[:100])
        self.assertTrue(np.allclose(W.mean(0), 1 / V, atol=0.05))
        self.assertLessEqual(alert.mean(), 0.05)

    def test_corrupted_view_downweighted(self):
        rc = ReliabilityCalib(calib())
        M = calib(50)
        M[:, 1] *= 50.0                        # view 1 heavily corrupted
        W, _ = rc.weights(M)
        self.assertTrue((W[:, 1] < W[:, 0]).all())
        self.assertTrue((W[:, 1] < W[:, 2]).all())

    def test_scale_free_across_views(self):
        # calib-median normalisation removes per-view MSE scale: a view with
        # 10x the raw scale must not be systematically downweighted
        M = calib(2000, scale=(1.0, 10.0, 1.0))
        rc = ReliabilityCalib(M)
        W, _ = rc.weights(M[:200])
        self.assertTrue(np.allclose(W.mean(0), 1 / V, atol=0.05))

    def test_alert_requires_all_views(self):
        Mc = calib(500)
        rc = ReliabilityCalib(Mc)
        one = np.median(Mc, 0)[None, :].repeat(3, 0)
        one[0, 0] *= 1e3                       # single view blown: no alert
        one[1] *= 1e3                          # all views blown: alert
        _, alert = rc.weights(one)
        self.assertFalse(alert[0])
        self.assertTrue(alert[1])
        self.assertFalse(alert[2])

    def test_clip_bounds_respected(self):
        rc = ReliabilityCalib(calib())
        M = calib(50)
        M[:, 0] *= 1e6                         # u collapses to clip floor
        W, _ = rc.weights(M)
        E = M / rc.med
        U = np.clip(1 / (E + 1e-8), rc.u_lo, rc.u_hi)
        self.assertTrue(np.allclose(W, U / U.sum(1, keepdims=True)))


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.D = RNG.gamma(2.0, 1.0, (10, 4, V))
        self.S = RNG.gamma(2.0, 1.0, (4, V)) + 0.5

    def test_uniform_equals_mean_of_per_view_r(self):
        R = aggregate(self.D, self.S, uniform_weights(10, V))
        self.assertTrue(np.allclose(R, per_view_r(self.D, self.S).mean(2)))

    def test_uniform_differs_from_legacy_ratio_of_means(self):
        legacy = self.D.mean(2) / (self.S.mean(1)[None] + 1e-6)
        R = aggregate(self.D, self.S, uniform_weights(10, V))
        self.assertFalse(np.allclose(R, legacy))   # the S0 control is real

    def test_oracle_mask_removes_polluted_view(self):
        Dp = self.D.copy()
        Dp[:, :, 1] += 100.0                   # pollute view 1 distances
        Rm = aggregate(Dp, self.S, oracle_mask_weights(10, V, 1))
        Rc = aggregate(self.D, self.S, oracle_mask_weights(10, V, 1))
        self.assertTrue(np.allclose(Rm, Rc))   # masked view cannot leak

    def test_weight_shift_moves_aggregate(self):
        W = uniform_weights(10, V)
        W2 = W.copy()
        W2[:, 0] = 0.6
        W2[:, 1:] = 0.2
        self.assertFalse(np.allclose(aggregate(self.D, self.S, W),
                                     aggregate(self.D, self.S, W2)))


if __name__ == "__main__":
    unittest.main()
