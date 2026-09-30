"""Protocol properties for source-only statistics and missing-class evaluation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_restored import expected_parts, metrics, quality_mask
from frozen_probe import fit_statistics, fit_readout, predict


class ProbeTests(unittest.TestCase):
    def test_missing_classes_do_not_lower_macro_denominator(self):
        score = metrics(np.array([0, 0, 2]), np.array([0, 1, 2]))
        self.assertAlmostEqual(score['macro_accuracy'], .75)
        self.assertAlmostEqual(score['accuracy'], 2/3)

    def test_whole_user_partitions_and_quality_exclusions(self):
        subjects = np.repeat(np.arange(1, 9), 12)
        y = np.tile(np.arange(12), 8)
        parts = expected_parts(y, subjects, 8)
        self.assertEqual(set(subjects[parts['train']]), set(range(2, 8)))
        self.assertEqual(set(subjects[parts['val']]), {1})
        self.assertEqual(set(subjects[parts['test']]), {8})
        ids = np.concatenate(list(parts.values()))
        np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(quality_mask(y, subjects)))
        self.assertEqual(len(ids), len(np.unique(ids)))

    def test_class_identical_users_have_zero_penalty(self):
        y = np.tile(np.arange(12), 3)
        subjects = np.repeat(np.arange(1, 4), 12)
        f = np.eye(12)[y]
        stats = fit_statistics(f, y, subjects)
        np.testing.assert_allclose(stats['drift_covariance'], 0, atol=1e-28)
        w, b = fit_readout(stats, .01, 10)
        np.testing.assert_array_equal(predict(f, stats, w, b).argmax(-1), y)

    def test_stats_order_invariant_with_missing_user_class_cell(self):
        rng = np.random.default_rng(2026)
        y = np.tile(np.arange(12), 3)
        subjects = np.repeat(np.arange(1, 4), 12)
        keep = ~((y == 3) & (subjects == 2))
        y, subjects = y[keep], subjects[keep]
        f = rng.normal(size=(len(y), 8))
        stats = fit_statistics(f, y, subjects)
        order = rng.permutation(len(y))
        shuffled = fit_statistics(f[order], y[order], subjects[order])
        for key in stats:
            np.testing.assert_allclose(stats[key], shuffled[key], atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(stats['drift_covariance']).min(), -1e-12)
        before = {k: v.copy() for k, v in stats.items()}
        w, b = fit_readout(stats, .1, 1.)
        predict(rng.normal(size=(7, 8)), stats, w, b)
        for key in stats:
            np.testing.assert_array_equal(stats[key], before[key])


if __name__ == '__main__':
    unittest.main()
