import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,resource,pickle,signal
from guard import ready,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
    p=ready();out=B/'profile01';out.mkdir(exist_ok=False)
    (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,real_UWB_values_labels_fit=False),indent=2)+'\n')
    signal.alarm(610);start=time.monotonic()
    sys.path.insert(0,'/workspace/uwb-rocket-source-probe-20261102')
    from native_rocket import np,make,assert_native
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    g=np.random.RandomState(p['seed']);x=g.randn(64,567,90).astype('float32')
    # Warm compiler and exact full default-size native pipeline on synthetic data.
    m=make();m.fit(x,np.arange(64)%12);assert_native(m,True)
    transform=[]
    for i in range(3):
        t=time.monotonic();f=m._transformer.transform(x);transform.append(time.monotonic()-t)
        assert f.shape==(64,20000) and np.isfinite(f).all()
    # Full training shape: no cubic RidgeCV extrapolation from tiny N.
    F=g.standard_normal((4700,20000)).astype('float32');F[:,::2]=np.clip(F[:,::2]/6+.5,0,1)
    readout=make_pipeline(StandardScaler(with_mean=False),RidgeClassifierCV(alphas=np.logspace(-3,3,10)))
    t=time.monotonic();readout.fit(F,np.arange(4700)%12);ridge=time.monotonic()-t
    assert readout[-1].coef_.shape==(12,20000)
    m.pipeline_.steps[1]=(m.pipeline_.steps[1][0],readout[0]);m.pipeline_.steps[2]=(m.pipeline_.steps[2][0],readout[1]);m.estimator_=readout[1];m._scaler=readout[0]
    saves=[]
    for i in range(3):
        t=time.monotonic()
        with (out/f'native{i}.pkl').open('wb') as h:pickle.dump(m,h,protocol=5)
        saves.append(time.monotonic()-t)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r=select(p,transform,ridge,saves,rss);r.update(policy=p,transform64_seconds=transform,
        full_RidgeCV_seconds=ridge,native_save_seconds=saves,max_RSS_KiB=rss,
        wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,
        real_UWB_values_labels_fit=False,profile_completed_successfully=True)
    (out/'COST_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
