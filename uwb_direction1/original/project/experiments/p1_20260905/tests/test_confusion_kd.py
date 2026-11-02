import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confusion_kd import projected_kd, uniform_control, inner_splits, estimate_matrix
from train import kd_loss


class ConfusionKDTests(unittest.TestCase):
    def test_identity_reduces_to_ordinary_kd(self):
        torch.manual_seed(1)
        s = torch.randn(5, 12); t = torch.randn(5, 12)
        torch.testing.assert_close(projected_kd(s, t, torch.eye(12)), kd_loss(s, t))

    def test_targets_detach_and_student_gradient(self):
        s = torch.randn(5, 12, requires_grad=True)
        t = torch.randn(5, 12, requires_grad=True)
        matrix = torch.eye(12, requires_grad=True)
        projected_kd(s, t, matrix).backward()
        self.assertIsNone(t.grad); self.assertIsNone(matrix.grad)
        self.assertGreater(s.grad.abs().sum().item(), 0)

    def test_uniform_preserves_diagonal_and_probability(self):
        matrix = np.array([[.7, .25, .05], [.1, .6, .3], [.02, .08, .9]])
        u = uniform_control(matrix)
        np.testing.assert_allclose(u.sum(1), 1)
        np.testing.assert_allclose(u.diagonal(), matrix.diagonal())
        self.assertAlmostEqual(u[0, 1], u[0, 2])

    def test_inner_folds_never_touch_outer_reserved_ids(self):
        subjects = np.repeat(np.arange(1, 5), 10)
        tr = np.array([i for i in range(30) if i % 10 != 9])
        held = []
        for train_ids, held_ids in inner_splits(tr, subjects):
            self.assertFalse(set(subjects[train_ids]) & set(subjects[held_ids]))
            self.assertEqual(set(train_ids) | set(held_ids), set(tr))
            held.extend(held_ids)
        np.testing.assert_array_equal(np.sort(held), tr)

    def test_matrix_uses_only_supplied_training_rows(self):
        p = np.array([[.9, .1], [.2, .8], [.4, .6], [np.nan, np.nan]])
        y = np.array([0, 1, 0, 1]); ids = np.array([0, 1, 2])
        c = estimate_matrix(p, y, ids)
        np.testing.assert_allclose(c, [[.65, .35], [.2, .8]])
        np.testing.assert_allclose(c.sum(1), 1)


if __name__ == '__main__': unittest.main()
