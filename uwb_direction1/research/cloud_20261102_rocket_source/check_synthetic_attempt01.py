"""Synthetic-only full-size implementation checks; no UWB path access."""
from pathlib import Path
import json,pickle,hashlib,sys
from native_rocket import make,assert_native,numpy_features,np
B=Path(__file__).resolve().parent
def main():
    g=np.random.RandomState(20261029);x=g.randn(24,567,90).astype('float32')
    x[:,0]=0;y=np.arange(24)%12
    m=make();m.fit(x,y);assert_native(m,True)
    t,s,r=[v for _,v in m.pipeline_.steps]
    f=t.transform(x);assert f.shape==(24,20000) and f.dtype==np.float32
    which=list(range(32))+[4999,9999];cols=np.array([[2*j,2*j+1] for j in which]).ravel()
    maxerr=0.
    for i in range(2):
        q=numpy_features(x[i],t.kernels,which)
        np.testing.assert_allclose(f[i,cols],q,rtol=1e-5,atol=1e-5)
        np.testing.assert_array_equal(f[i,cols[::2]],q[::2]);maxerr=max(maxerr,float(abs(f[i,cols]-q).max()))
    # Train-only scaler and independent analytic LOO including intercept leverage.
    np.testing.assert_allclose(s.mean_,f.mean(0,dtype=np.float64),atol=1e-10)
    Z=s.transform(f).astype(np.float64);Z-=Z.mean(0)
    Y=2*(y[:,None]==np.arange(12)).astype(float)-1;Y-=Y.mean(0)
    K=Z@Z.T;scores=[]
    for alpha in r.alphas:
        H=K@np.linalg.inv(K+alpha*np.eye(len(y)))+np.ones((len(y),len(y)))/len(y)
        err=(Y-H@Y)/(1-np.diag(H))[:,None];scores.append(-float(np.mean(err**2)))
    assert float(r.alpha_)==float(r.alphas[np.argmax(scores)])
    np.testing.assert_allclose(r.best_score_,max(scores),rtol=2e-4,atol=2e-5)
    before=[a.copy() for a in t.kernels];scale=s.scale_.copy()
    v=g.randn(5,567,90).astype('float32');p=m.predict(v)
    for a,b in zip(before,t.kernels):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(scale,s.scale_)
    raw=pickle.dumps(m,protocol=5);reload=pickle.loads(raw)
    np.testing.assert_array_equal(p,reload.predict(v))
    q=make();q.fit(x,y)
    for a,b in zip(t.kernels,q._transformer.kernels):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(m.predict(x),q.predict(x))
    report=dict(state='pass',synthetic_only=True,UWB_values_labels_or_fit=False,
       full_kernels=10000,features=20000,native_train_only_RidgeCV=True,
       independent_kernel_checks=68,independent_max_error=maxerr,
       constant_channel_normalization=True,independent_LOO_alpha_match=True,
       train_only_scaler_immutable_during_prediction=True,
       pickle_reload_predictions_exact=True,seed_repeat_exact=True,
       model_bytes=len(raw),synthetic_alpha=float(r.alpha_))
    (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    # Freeze actual imported executable/Python scientific closure, not whole OS.
    files={}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):
            files[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=files,scope='Imported scientific Python and binary closure, not entire OS'),indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
