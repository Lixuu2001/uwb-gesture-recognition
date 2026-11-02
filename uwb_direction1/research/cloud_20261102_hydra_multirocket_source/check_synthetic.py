from pathlib import Path
import json,pickle,hashlib,sys
from native_hydramr import np,torch,make,assert_native,components,independent_first_group
B=Path(__file__).resolve().parent
def main():
    g=np.random.RandomState(20261029);x=g.randn(24,567,90).astype('float32');y=np.arange(24)%12
    m=make();m.fit(x,y);assert_native(m,True)
    rawh,rawmr,F=components(m,x);assert F.shape==(24,53824) and np.isfinite(F).all()
    # SparseScaler independently reconstructed, including unbiased std and zero mask.
    q=np.sqrt(np.maximum(rawh.numpy(),0));eps=np.mean(q==0,axis=0,dtype=np.float32)**4+1e-8
    mu=q.mean(0);sigma=q.std(0,ddof=1)+eps
    expected=((q-mu)*(q!=0))/sigma
    np.testing.assert_allclose(m._scale_hydra.transform(rawh).numpy(),expected,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(m._scale_multirocket.mean_,rawmr.mean(0,dtype=np.float64),atol=1e-10)
    # Full default shape. Independent convolution/counting for first group of each branch/dilation.
    v=g.randn(3,567,90).astype('float32');vh,vm,vf=components(m,v)
    columns=np.concatenate([np.arange(block*256,block*256+8) for block in range(16)])
    for i in range(3):
        ref=independent_first_group(v[i],m._transform_hydra._hydra)
        np.testing.assert_allclose(vh[i].numpy()[columns],ref,rtol=2e-5,atol=2e-5)
        np.testing.assert_array_equal(vh[i].numpy()[columns.reshape(16,8)[1::2].ravel()],ref.reshape(16,8)[1::2].ravel())
    # Stable independent native LOO selection after registered full-training scaling.
    Z=F.astype(np.float64);Z-=Z.mean(0);K=Z@Z.T
    Y=2*(y[:,None]==np.arange(12))-1;Y=Y-Y.mean(0)
    C=np.vstack([np.eye(23),-np.ones((1,23))]);Q=np.linalg.qr(C)[0];scores=[]
    for alpha in m.classifier.alphas:
        A=Q@np.linalg.inv(Q.T@K@Q+alpha*np.eye(23))@Q.T;err=(A@Y)/np.diag(A)[:,None];scores.append(-float(np.mean(err**2)))
    assert m.classifier.alpha_==m.classifier.alphas[np.argmax(scores)]
    np.testing.assert_allclose(m.classifier.best_score_,max(scores),rtol=2e-4,atol=2e-5)
    before=pickle.dumps(m,protocol=5);p=m.predict(v);after=pickle.dumps(m,protocol=5)
    # Torch pickle storage IDs can vary between serializations: compare scientific arrays explicitly.
    restored=pickle.loads(before);np.testing.assert_array_equal(p,restored.predict(v))
    np.testing.assert_array_equal(p,m.classifier.predict(vf));second=make();second.fit(x,y)
    np.testing.assert_array_equal(m._transform_hydra._hydra.W.numpy(),second._transform_hydra._hydra.W.numpy())
    np.testing.assert_array_equal(m.classifier.coef_,second.classifier.coef_)
    np.testing.assert_array_equal(p,second.predict(v))
    report=dict(state='pass',synthetic_only=True,full_default_features=53824,Hydra_features=4096,MultiRocket_features=49728,groups=64,kernels_per_group=8,independent_Hydra_scalar_counts_checked=384,independent_SparseScaler_formula=True,native_MultiRocket_raw_core_finite_before_fallback=True,independent_LOO_alpha_match=True,seed_repeat_exact=True,pickle_reload_and_staged_predictions_exact=True,UWB_values_labels_fit=False)
    (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    files={}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):files[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=files,scope='Imported scientific Python/binary closure,not entire OS'),indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
