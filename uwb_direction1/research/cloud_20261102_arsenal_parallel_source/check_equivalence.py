"""New CPU4 execution validation only; old frozen tests/profiles untouched."""
from pathlib import Path
import json,hashlib,pickle,sys
from native_arsenal import np,make,assert_native,canonical_predictions,Arsenal
B=Path(__file__).resolve().parent
def arrays(model):
    for pipe in model.estimators_:
        t,s,r=(v for _,v in pipe.steps)
        for a in [*t.kernels,s.mean_,s.var_,s.scale_,r.coef_,r.intercept_,r.classes_,r.alpha_,r.best_score_]:yield np.asarray(a)
    yield np.asarray(model.weights_);yield np.asarray(model._weight_sum)
def main():
    rng=np.random.RandomState(20261029);y=np.arange(24)%12;t=np.linspace(0,2*np.pi,90,endpoint=False,dtype=np.float32)
    x=(.15*rng.randn(24,567,90)+3*np.sin((y[:,None,None]+1)*t[None,None,:])).astype('float32');x[:,0]=0
    parallel=make();parallel.fit(x,y);assert_native(parallel,True)
    serial=Arsenal(n_kernels=2000,n_estimators=25,rocket_transform='rocket',time_limit_in_minutes=0,class_weight=None,n_jobs=1,random_state=20261029,verbose=0);serial.fit(x,y)
    a=list(arrays(serial));b=list(arrays(parallel));assert len(a)==len(b)
    for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
    p=parallel.predict_proba(x);np.testing.assert_array_equal(p,serial.predict_proba(x));np.testing.assert_array_equal(parallel.predict(x),serial.predict(x))
    np.testing.assert_array_equal(parallel.predict(x),canonical_predictions(parallel,p))
    staged=np.concatenate([parallel.predict_proba(np.ascontiguousarray(x[i:i+7])) for i in range(0,24,7)]);np.testing.assert_array_equal(staged,p)
    saved=pickle.dumps(parallel,protocol=5);q=pickle.loads(saved);np.testing.assert_array_equal(q.predict_proba(x),p)
    for left,right in zip(arrays(q),b):np.testing.assert_array_equal(left,right)
    report=dict(state='passed_full25x2000_native_jobs1_vs_jobs4_execution_equivalence_on_synthetic_fixture',cases=24,channels=567,timepoints=90,classes=12,members=25,kernels_per_member=2000,total_kernels=50000,seed=20261029,native_parallel_jobs=4,BLAS_threads_per_job=1,Rocket_transform_threads_per_member=1,Numba_parallel_lock_retained=True,all_kernel_scaler_readout_alpha_LOOaccuracy_weight_arrays_exact=True,native_probabilities_canonical_tie_predictions_exact=True,staged_probabilities_exact=True,pickle_reload_exact=True,model_bytes=len(saved),actual_UWB_values_labels_fit=False,accuracy_or_novelty_claim=False)
    (B/'EQUIVALENCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    deps={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):deps[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for pth in B.iterdir():
        if pth.is_file():deps[str(pth)]=hashlib.sha256(pth.read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=deps,scope='Imported native scientific Python/binary closure and new parallel execution prep;notwholeOS'),indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
