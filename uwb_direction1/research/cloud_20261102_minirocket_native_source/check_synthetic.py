from pathlib import Path
import json,hashlib,pickle,sys
from native_minirocket import np,make,assert_native,independent_ppv
B=Path(__file__).resolve().parent
def main():
    g=np.random.RandomState(20261029);x=g.randn(24,567,90).astype('float32');y=np.arange(24)%12
    m=make();m.fit(x,y);assert_native(m,True);t=m._transformer;f=t.transform(x)
    assert f.shape==(24,9996) and np.isfinite(f).all()
    v=g.randn(4,567,90).astype('float32');vf=t.transform(v);which=[0,1,10,99,1000,9995]
    for i in range(4):np.testing.assert_array_equal(vf[i,which],independent_ppv(v[i],t,which))
    Z=m._scaler.transform(f).astype(np.float64);Z-=Z.mean(0);K=Z@Z.T
    Y=2*(y[:,None]==np.arange(12))-1;Y=Y-Y.mean(0)
    C=np.vstack([np.eye(23),-np.ones((1,23))]);Q=np.linalg.qr(C)[0];scores=[]
    for alpha in m.estimator_.alphas:
        A=Q@np.linalg.inv(Q.T@K@Q+alpha*np.eye(23))@Q.T;err=(A@Y)/np.diag(A)[:,None];scores.append(-float(np.mean(err**2)))
    assert m.estimator_.alpha_==m.estimator_.alphas[np.argmax(scores)]
    np.testing.assert_allclose(m.estimator_.best_score_,max(scores),rtol=2e-4,atol=2e-5)
    before=[a.copy() for a in t.parameters];s=m._scaler.scale_.copy();p=m.predict(v)
    for a,b in zip(before,t.parameters):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(s,m._scaler.scale_)
    reload=pickle.loads(pickle.dumps(m,protocol=5));np.testing.assert_array_equal(p,reload.predict(v))
    second=make();second.fit(x,y)
    for a,b in zip(t.parameters,second._transformer.parameters):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(m.predict(v),second.predict(v))
    report=dict(state='pass',synthetic_only=True,full_default_features=9996,requested_n_kernels=10000,max_dilations_per_kernel=32,seed=20261029,independent_PPVs_checked=24,independent_LOO_alpha_match=True,seed_repeat_exact=True,pickle_reload_predictions_exact=True,prediction_preserves_train_only_transform_and_scaler=True,UWB_values_labels_fit=False)
    (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    files={}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):files[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=files,scope='Imported scientific source/binary closure,not whole OS'),indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
