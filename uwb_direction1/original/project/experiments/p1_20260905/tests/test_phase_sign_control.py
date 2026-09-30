import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase_sign_control import unsigned_from_motion, unsigned_phase
from pipeline import norm, doppler_map


class SignControlTests(unittest.TestCase):
    def test_removes_sign_before_standardization(self):
        z = np.array([[[1., -2., 4.], [-6., 8., -1.]]], dtype=np.float32)
        a = unsigned_from_motion(z)
        np.testing.assert_allclose(a, norm(np.abs(z)))
        np.testing.assert_array_equal(a, unsigned_from_motion(-z))
        self.assertFalse(np.allclose(a, norm(np.abs(norm(z)))))

    def test_physical_pipeline_and_shape(self):
        x = np.random.default_rng(21).normal(size=(1, 90, 189)).astype('float32')
        z = unsigned_phase(x)
        self.assertEqual(z.shape, x.shape)
        np.testing.assert_allclose(z, norm(np.abs(doppler_map(x))))
        self.assertTrue(np.isfinite(z).all())


if __name__ == '__main__': unittest.main()
