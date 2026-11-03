import unittest,math
import numpy as np
from audit import independent_mixture
from vote import combine
class ScalarRepair(unittest.TestCase):
    def test_float32_numpy2_scalar_subtraction_requires_explicit_double(self):
        rng=np.random.default_rng(20261029)
        a=rng.normal(0,8,(64,12)).astype(np.float32)
        b=rng.normal(0,8,(64,12)).astype(np.float32)
        ids=np.arange(64);q,_=combine(ids,a,ids,b)
        actual=independent_mixture(a,b)
        np.testing.assert_allclose(actual,q,rtol=0,atol=1e-14)
        # Demonstrate the old reference's loss of precision on synthetic inputs.
        row=b[0];old=np.array([math.exp(float(x)-max(row)) for x in row]);old/=old.sum()
        fresh=np.array([math.exp(float(x)-float(max(row))) for x in row]);fresh/=fresh.sum()
        self.assertGreater(float(np.max(np.abs(old-fresh))),1e-14)
if __name__=='__main__':unittest.main(verbosity=2)
