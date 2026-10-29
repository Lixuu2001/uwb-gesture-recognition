"""Bounded compatibility/determinism/numerics checks; never opens UWB files."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib, json, pickle, resource, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from adapter import FixedMultiRocket, fit_readout, logits, balanced_weights

def main():
    resource.setrlimit(resource.RLIMIT_CPU, (580, 590))
    start = time.monotonic()
    rng = np.random.default_rng(123)
    x = rng.normal(size=(16, 567, 90)).astype('float32')
    a = FixedMultiRocket().fit(x)
    b = FixedMultiRocket().fit(x)
    for p,q in zip(a.model.parameter,b.model.parameter):np.testing.assert_array_equal(p,q)
    for p,q in zip(a.model.parameter1,b.model.parameter1):np.testing.assert_array_equal(p,q)
    t = time.monotonic();f = a.transform(x[:4]);seconds = time.monotonic()-t
    t=time.monotonic();repeated=a.transform(x[:4]);warm_seconds=time.monotonic()-t
    np.testing.assert_array_equal(f,repeated)
    np.testing.assert_array_equal(f, np.concatenate([a.transform(x[:2]), a.transform(x[2:4])]))
    np.testing.assert_array_equal(f, a.model.transform(x[:4]))
    c = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(f, c.transform(x[:4]))
    toy = rng.normal(size=(80, 121))
    y = np.arange(80) % 4;users = np.arange(80) % 3
    r = fit_readout(toy, y, users)
    sk = RidgeClassifier(alpha=1., solver='cholesky').fit(r['scaler'].transform(toy),y,
        sample_weight=balanced_weights(y,users))
    np.testing.assert_allclose(logits(r,toy),sk.decision_function(r['scaler'].transform(toy)),atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(r['classes'][logits(r,toy).argmax(1)],sk.predict(r['scaler'].transform(toy)))
    result=dict(state='pass',synthetic_only=True,uwb_arrays_or_labels_read=False,
        shape=list(f.shape),full_size_features=49728,deterministic_parameters=True,
        batch_invariant=True,public_api_parity=True,pickle_replay_exact=True,
        weighted_dual_ridge_matches_sklearn=True,
        normal_equation_relative_residual=r['normal_equation_relative_residual'],
        transform4_including_compilation_seconds=seconds,
        warm_transform4_seconds=warm_seconds,
        aeon_version=__import__('aeon').__version__,numba_version=__import__('numba').__version__,
        llvmlite_version=__import__('llvmlite').__version__,
        wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name('adapter.py').read_bytes()).hexdigest())
    out=Path(__file__).with_name('SYNTHETIC_CHECK.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
