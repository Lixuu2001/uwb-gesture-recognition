from pathlib import Path
import json,pickle,hashlib,sys
from native_arsenal import make,assert_native,canonical_predictions,np
from aeon.transformations.collection import Normalizer
B=Path(__file__).resolve().parent
def independent_LOO_accuracy(Z,y,alphas):
    Z=Z.astype(np.float64);Z-=Z.mean(0);Y=2*(y[:,None]==np.arange(12)).astype(float)-1
    K=Z@Z.T;C=np.vstack([np.eye(len(y)-1),-np.ones((1,len(y)-1))]);Q=np.linalg.qr(C)[0];centered=Y-Y.mean(0);scores=[]
    for alpha in alphas:
        A=Q@np.linalg.inv(Q.T@K@Q+float(alpha)*np.eye(len(y)-1))@Q.T
        residual=(A@centered)/np.diag(A)[:,None]
        scores.append(float(np.mean(np.argmax(Y-residual,axis=1)==y)))
    return np.asarray(scores)
def main():
    rng=np.random.RandomState(20261029);y=np.arange(24)%12
    # Two samples/class pure noise has zero LOO vote weight for all members;
    # use a deterministic synthetic learnable signal,never actual UWB labels.
    phase=np.linspace(0,2*np.pi,90,endpoint=False,dtype=np.float32)
    x=(.15*rng.randn(24,567,90)+3*np.sin((y[:,None,None]+1)*phase[None,None,:])).astype('float32');x[:,0]=0
    m=make();m.fit(x,y);assert_native(m,True);z=Normalizer().fit_transform(x).astype('float32',copy=False)
    scores=[];votes=np.zeros((len(y),12));weights=[]
    for pipe,w in zip(m.estimators_,m.weights_):
        t,s,r=[v for _,v in pipe.steps];f=t._transform_kernels(z);assert f.shape==(24,4000) and np.isfinite(f).all()
        np.testing.assert_allclose(s.mean_,f.mean(0,dtype=np.float64),atol=1e-10)
        cv=independent_LOO_accuracy(s.transform(f),y,r.alphas)
        assert float(r.best_score_)==float(cv.max())
        # Tied alpha scores may use numerically different but equally best values;
        # require selected alpha's independent score to equal optimum.
        assert cv[list(r.alphas).index(r.alpha_)]==cv.max()
        pred=r.predict(s.transform(f));votes[np.arange(len(y)),pred]+=w;weights.append(w);scores.append(float(r.best_score_))
    expected= votes/float(np.sum(weights));np.testing.assert_array_equal(m.predict_proba(x),expected)
    np.testing.assert_array_equal(m.predict(x),canonical_predictions(m,expected))
    # Synthetic tie-only fixture exercises exact native RNG draw order.
    ties=np.ones((9,12))/12;g=np.random.RandomState(20261029);ref=np.array([m.classes_[g.choice(np.arange(12))] for _ in range(9)])
    np.testing.assert_array_equal(canonical_predictions(m,ties),ref)
    v=rng.randn(7,567,90).astype('float32');p=m.predict_proba(v)
    staged=np.concatenate([m.predict_proba(np.ascontiguousarray(v[i:i+3])) for i in range(0,len(v),3)])
    np.testing.assert_array_equal(staged,p)
    raw=pickle.dumps(m,protocol=5);reload=pickle.loads(raw);np.testing.assert_array_equal(reload.predict_proba(v),p)
    q=make();q.fit(x,y);assert_native(q,True)
    for a,b in zip(m.estimators_,q.estimators_):
        for w1,w2 in zip(a.steps[0][1].kernels,b.steps[0][1].kernels):np.testing.assert_array_equal(w1,w2)
        for key in ['coef_','intercept_','alpha_','best_score_']:np.testing.assert_array_equal(getattr(a.steps[2][1],key),getattr(b.steps[2][1],key))
    np.testing.assert_array_equal(q.predict_proba(v),p)
    report=dict(state='passed_full25x2000_native_Arsenal_synthetic_checks',cases=24,channels=567,timepoints=90,classes=12,members=25,kernels_per_member=2000,total_kernels=50000,features_per_member=4000,train_only_native_LOO_accuracy_weighting=True,independent_all25_LOO_optimum_match=True,independent_all25_scaler_mean=True,independent_weighted_vote_exact=True,canonical_tie_RNG_exact=True,staged_probabilities_exact=True,pickle_reload_exact=True,seed_repeat_allkernels_and_readouts_exact=True,model_bytes=len(raw),actual_UWB_values_labels_fit=False,novelty_claim=False)
    (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    deps={}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and (name.endswith('.py') or '.so' in name):deps[str(Path(name).resolve())]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for p in B.iterdir():
        if p.is_file() and p.suffix=='.py':deps[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(scope='Imported scientific Python/binary closure,not entireOS',files=deps),indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
