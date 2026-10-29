"""Fixed full-size MultiRocket transform and a checked weighted ridge readout.

No UWB data are read here. The aeon kernels are used without local edits.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / 'vendor/site-packages'))
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler
from numba import set_num_threads

from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._multirocket import _transform_multi

class FixedMultiRocket:
    def __init__(self, seed=20261029):
        self.seed = seed
        self.model = MultiRocket(n_kernels=6250, max_dilations_per_kernel=32,
            n_features_per_kernel=4, normalise=False, n_jobs=1, random_state=seed)

    def fit(self, x):
        set_num_threads(1)
        self.model.fit(np.ascontiguousarray(x, dtype=np.float32))
        return self

    def transform(self, x):
        set_num_threads(1)
        x = np.ascontiguousarray(x, dtype=np.float32)
        # Inspect core output before the public API's np.nan_to_num fallback.
        f = _transform_multi(x, np.diff(x, axis=2), self.model.parameter,
            self.model.parameter1, 4, MultiRocket._indices, self.model.random_state_)
        assert f.shape == (len(x), 49728)
        assert np.isfinite(f).all(), 'nonfinite transform; do not silently fill'
        return f

def balanced_weights(y, users):
    w = np.zeros(len(y), dtype=np.float64)
    for u in np.unique(users):
        for c in np.unique(y[users == u]):
            mask = (users == u) & (y == c)
            w[mask] = 1 / mask.sum()
    return w * (len(w) / w.sum())

def fit_readout(features, y, users):
    """Alpha=1, unweighted training scaler, weighted intercept, exact dual solve."""
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(features, dtype=np.float64))
    classes = np.unique(y)
    assert len(classes) > 2
    target = np.where(y[:, None] == classes[None, :], 1., -1.)
    w = balanced_weights(y, users)
    xm = np.average(x, axis=0, weights=w)
    ym = np.average(target, axis=0, weights=w)
    z = (x - xm) * np.sqrt(w[:, None])
    rhs = (target - ym) * np.sqrt(w[:, None])
    gram = z @ z.T
    gram.flat[::len(gram) + 1] += 1.
    dual = cho_solve(cho_factor(gram, lower=True, check_finite=True), rhs)
    residual = float(np.linalg.norm(gram @ dual - rhs) / max(np.linalg.norm(rhs), 1e-30))
    assert residual < 1e-8, 'linear solve incomplete; not a scientific negative'
    coef = z.T @ dual
    intercept = ym - xm @ coef
    return dict(scaler=scaler, classes=classes, coef=coef, intercept=intercept,
        normal_equation_relative_residual=residual, alpha=1., weights=w)

def logits(readout, features):
    return readout['scaler'].transform(np.asarray(features, dtype=np.float64)) @ readout['coef'] + readout['intercept']
