"""GPL-3.0-compatible test wrapper;unchanged original GPL source local-only.

Feature extraction preparation only. Arsenal remains the active real study.
No classifier fitting,resource timing,method selection or UWB data access.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,importlib.util,json,hashlib,math
import numpy as np
import torch
import cloudpickle
torch.set_num_threads(1);torch.set_num_interop_threads(1)
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-quant-static-review-20261102');SRC=S/'local_author_sources/code/quant.py'

def reference_intervals(length,depth=6):
    intervals=[]
    for level in range(min(depth,int(math.log2(length))+1)):
        n=2**level;v=[(int(length*i/n),int(length*(i+1)/n)) for i in range(n)];intervals.extend(v)
        sizes=sorted(b-a for a,b in v)
        if n>1 and sizes[(len(sizes)-1)//2]>1:
            shift=math.ceil(length/n/2);intervals.extend([(a+shift,b+shift) for a,b in v[:-1]])
    return np.asarray(intervals,dtype=np.int64)

def reference_features(x):
    # Independent NumPy/FP64 representations,replicatepad and linearquantiles.
    a=x.astype(np.float64);d=np.diff(a,axis=-1);p=np.pad(d,((0,0),(0,0),(2,2)),mode='edge')
    smooth=sum(p[...,i:i+d.shape[-1]] for i in range(5))/5
    reps=[a,smooth,np.diff(a,n=2,axis=-1),np.abs(np.fft.rfft(a,axis=-1))];parts=[]
    for z in reps:
        for start,end in reference_intervals(z.shape[-1]):
            window=z[...,start:end];n=window.shape[-1];q=1+(n-1)//4
            if n==1:f=window
            else:
                grid=np.asarray([.5]) if q==1 else np.linspace(0,1,q)
                f=np.quantile(window,grid,axis=-1,method='linear').transpose(1,2,0)
                if q>1:f[...,1::2]-=window.mean(-1,keepdims=True)
            parts.append(f.reshape(len(x),-1))
    return np.concatenate(parts,axis=-1)

def main():
    provenance=json.loads((S/'STATIC_REVIEW.json').read_bytes());row=next(r for r in provenance['upstream_files'] if r['path']=='code/quant.py');assert hashlib.sha256(SRC.read_bytes()).hexdigest()==row['sha256']
    spec=importlib.util.spec_from_file_location('author_quant',SRC);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    rng=np.random.default_rng(20261029);x=rng.standard_normal((3,3,90),dtype=np.float32);x[:,1]=0
    model=module.Quant();f=model.fit_transform(torch.from_numpy(x),np.zeros(3,dtype=np.int64)).numpy()
    assert f.shape==(3,3*951) and np.isfinite(f).all()
    ref=reference_features(x);np.testing.assert_allclose(f,ref,rtol=1e-5,atol=2e-6)
    for i,length in enumerate([90,89,88,46]):np.testing.assert_array_equal(model.models[i].intervals.numpy(),reference_intervals(length))
    staged=np.concatenate([model.transform(torch.from_numpy(x[i:i+1])).numpy() for i in range(3)]);np.testing.assert_array_equal(f,staged)
    clone=cloudpickle.loads(cloudpickle.dumps(model,protocol=5));np.testing.assert_array_equal(clone.transform(torch.from_numpy(x)).numpy(),f)
    # Full target width check remains a tiny synthetic N=2 fixture;no classifier.
    full=np.tile(x[:2],(1,189,1));wide=module.Quant();a=wide.fit_transform(torch.from_numpy(full),np.zeros(2,dtype=np.int64)).numpy();assert a.shape==(2,539217) and np.isfinite(a).all()
    for i,length in enumerate([90,89,88,46]):np.testing.assert_array_equal(wide.models[i].intervals.numpy(),reference_intervals(length))
    # Verify feature concatenation under multivariate handling,not channelpruning.
    q=reference_features(full);np.testing.assert_allclose(a,q,rtol=1e-5,atol=2e-6)
    assert hashlib.sha256(SRC.read_bytes()).hexdigest()==row['sha256']
    report=dict(state='passed_unchanged_current_author_QUANT_multivariate_feature_preparation',author_repository='angus924/quant',author_commit=provenance['commit'],author_source_sha256=row['sha256'],GPL_original_unmodified_local_only=True,defaults=dict(depth=6,div=4),fixture_shape=list(x.shape),full_target_width_fixture_shape=list(full.shape),features_per_channel=951,full567channel_features=539217,independent_FP64_numpy_all_feature_reference_passed=True,reference_rtol=1e-5,reference_atol=2e-6,maximum_absolute_error=float(abs(f-ref).max()),full_width_maximum_absolute_error=float(abs(a-q).max()),interval_bounds_all_representations_exact=True,staged_native_features_exact=True,cloudpickle_transform_roundtrip_exact=True,previous_static_view_concern_not_reproduced=True,no_view_or_algorithm_patch=True,ExtraTrees200_not_fitted=True,formal_cost_not_started=True,actual_UWB_values_labels_fit=False,method_not_selected=True,accuracy_or_novelty_claim=False,priority='Arsenalrealstudyandfullaudit remainsactive')
    (B/'AUTHOR_FEATURE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
