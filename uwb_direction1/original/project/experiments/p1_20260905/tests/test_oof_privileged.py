import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import oof_privileged
except ImportError:
    oof_privileged = None


class OOFPrivilegedTests(unittest.TestCase):
    def test_cross_user_oof_covers_train_once_and_excludes_reserved(self):
        self.assertIsNotNone(oof_privileged, 'OOF privileged module is not implemented')
        rng = np.random.default_rng(7)
        subjects = np.repeat(np.arange(1, 6), 8)
        y = np.tile(np.repeat([0, 1], 4), 5)
        x = np.c_[y*3 + rng.normal(0, .1, len(y)), rng.normal(size=len(y))]
        train_ids = np.flatnonzero(subjects <= 4)
        reserved_ids = np.flatnonzero(subjects == 5)
        probabilities, history = oof_privileged.cross_user_oof(
            x, y, subjects, train_ids, reserved_ids, seed=2026)
        self.assertTrue(np.isfinite(probabilities[train_ids]).all())
        self.assertTrue(np.isnan(probabilities[reserved_ids]).all())
        np.testing.assert_allclose(probabilities[train_ids].sum(1), 1, atol=1e-7)
        predicted = np.concatenate([np.asarray(row['predict_ids']) for row in history])
        np.testing.assert_array_equal(np.sort(predicted), train_ids)
        self.assertEqual(len(np.unique(predicted)), len(train_ids))
        for row in history:
            self.assertFalse(np.intersect1d(row['fit_ids'], reserved_ids).size)
            self.assertFalse(np.intersect1d(row['predict_ids'], reserved_ids).size)
            self.assertFalse(np.intersect1d(row['fit_ids'], row['predict_ids']).size)

    def test_shape_target_fixes_true_mass_and_detaches_proxy(self):
        self.assertIsNotNone(oof_privileged, 'OOF privileged module is not implemented')
        labels = torch.tensor([0, 2])
        probabilities = torch.tensor([[.6, .3, .1], [.2, .1, .7]], requires_grad=True)
        self.assertTrue(hasattr(oof_privileged, 'shape_matched_target'),
                        'shape-matched target is not implemented')
        target = oof_privileged.shape_matched_target(labels, probabilities, .8)
        torch.testing.assert_close(target, torch.tensor([[.8, .15, .05],
                                                         [.13333334, .06666667, .8]]))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target.sum(1), torch.ones(2))

    def test_shape_loss_only_updates_student(self):
        self.assertIsNotNone(oof_privileged, 'OOF privileged module is not implemented')
        self.assertTrue(hasattr(oof_privileged, 'shape_matched_loss'),
                        'shape-matched loss is not implemented')
        student = torch.randn(3, 3, requires_grad=True)
        probabilities = torch.tensor([[.7, .2, .1], [.1, .8, .1], [.2, .2, .6]],
                                     requires_grad=True)
        loss = oof_privileged.shape_matched_loss(
            student, torch.tensor([0, 1, 2]), probabilities, .82)
        loss.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertIsNone(probabilities.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_pooled_features_keep_view_then_modality_order(self):
        self.assertIsNotNone(oof_privileged, 'OOF privileged module is not implemented')
        amplitude = np.zeros((2, 3, 90, 189), dtype=np.float32)
        phase = np.zeros_like(amplitude)
        for view in range(3):
            amplitude[:, view] = view + 1
            phase[:, view] = view + 11
        features = oof_privileged.pooled_features_from_arrays(amplitude, phase, [0, 2])
        self.assertEqual(features.shape, (2, 840))
        np.testing.assert_allclose(features[0, :210], 1)
        np.testing.assert_allclose(features[0, 210:420], 11)
        np.testing.assert_allclose(features[0, 420:630], 3)
        np.testing.assert_allclose(features[0, 630:], 13)


if __name__ == '__main__':
    unittest.main()
