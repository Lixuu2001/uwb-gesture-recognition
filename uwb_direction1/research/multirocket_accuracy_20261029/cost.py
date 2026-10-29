"""Synthetic-only single-thread estimate, independent of UWB outcomes."""
from adapter import FixedMultiRocket
import json,time,resource,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parent
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(280,290));start=time.monotonic()
 toy=np.random.default_rng(55).normal(size=(64,567,90)).astype('float32')
 t=time.monotonic();m=FixedMultiRocket().fit(toy);fitsec=time.monotonic()-t
 m.transform(toy[:1])
 t=time.monotonic();m.transform(toy);transformsec=time.monotonic()-t
 rng=np.random.default_rng(56);z=rng.normal(size=(1024,8192));rhs=rng.normal(size=(1024,12))
 t=time.monotonic();gram=z@z.T;gemm=time.monotonic()-t
 gram.flat[::1025]+=1.
 t=time.monotonic();dual=cho_solve(cho_factor(gram,lower=True),rhs);chol=time.monotonic()-t
 t=time.monotonic();coef=z.T@dual;projection=time.monotonic()-t
 assert np.isfinite(coef).all()
 n=4700/1024
 readout_estimates={str(f):gemm*n*n*(f/8192)+chol*n**3+projection*n*(f/8192) for f in [49728,9996]}
 # Include MiniRocket transform and IO at a conservative additional 180 seconds;
 # apply 3x overall margin before any scientific outcome is available.
 estimate=3*(fitsec+transformsec/64*6900+sum(readout_estimates.values())+180)
 result=dict(state='complete_synthetic_cost_only',synthetic_only=True,uwb_arrays_or_labels_read=False,
  single_thread=True,synthetic_fit_seconds=fitsec,warm_transform64_seconds=transformsec,
  matrix_shape=[1024,8192],synthetic_gram_seconds=gemm,synthetic_cholesky_seconds=chol,
  synthetic_projection_seconds=projection,scaled_readout_seconds=readout_estimates,
  additional_minirocket_transform_and_io_seconds=180,margin=3.,estimated_wall_seconds=estimate,
  wall_budget_seconds=1800,cpu_budget_seconds=1800,passed=bool(estimate<1700),
  wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
  adapter_sha256=hashlib.sha256((B/'adapter.py').read_bytes()).hexdigest())
 out=B/'COST.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
