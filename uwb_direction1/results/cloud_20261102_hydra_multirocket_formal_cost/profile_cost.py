import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,json,pickle,resource,signal
from guard import B,ready,sha
from cost_math import select
def main():
    p=ready();out=B/'profile01';out.mkdir(exist_ok=False)
    (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,real_UWB_values_labels_fit=False,full_training_shape=[4700,567,90]),indent=2)+'\n')
    signal.alarm(1190);start=time.monotonic();sys.path.insert(0,'/workspace/uwb-hydra-multirocket-source-probe-20261102')
    from native_hydramr import np,torch,make,assert_native,components
    # Full native fit, no small-N extrapolation of Hydra memory or RidgeCV complexity.
    rng=np.random.RandomState(p['seed']);x=rng.standard_normal((4700,567,90)).astype('float32');y=np.arange(4700)%12
    m=make();t=time.monotonic();m.fit(x,y);fit=time.monotonic()-t;assert_native(m,True)
    evaluation=[]
    for i in range(3):
        t=time.monotonic();_,_,f=components(m,x[i*64:(i+1)*64]);z=m.classifier.decision_function(f);pred=m.classifier.predict(f)
        np.testing.assert_array_equal(pred,m.classifier.classes_[z.argmax(1)]);evaluation.append(time.monotonic()-t)
    saves=[]
    for i in range(3):
        t=time.monotonic()
        with (out/f'native{i}.pkl').open('wb') as f:pickle.dump(m,f,protocol=5)
        saves.append(time.monotonic()-t)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r=select(p,fit,evaluation,saves,rss)
    r.update(policy=p,full_native_fit_seconds=fit,checked_evaluation64_seconds=evaluation,native_save_seconds=saves,max_RSS_KiB=rss,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,real_UWB_values_labels_fit=False,profile_completed_successfully=True)
    (out/'COST_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
