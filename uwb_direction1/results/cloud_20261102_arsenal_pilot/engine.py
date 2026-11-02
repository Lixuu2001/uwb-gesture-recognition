import sys,hashlib
sys.path.insert(0,'/workspace/uwb-arsenal-source-probe-20261102')
from native_arsenal import np,make,assert_native,canonical_predictions
from aeon.transformations.collection import Normalizer
def scientific_arrays(m):
    for pipe in m.estimators_:
        t,s,r=[v for _,v in pipe.steps]
        for a in [*t.kernels,s.mean_,s.var_,s.scale_,r.coef_,r.intercept_,r.classes_,np.asarray(r.alpha_),np.asarray(r.best_score_)]:yield np.asarray(a)
    yield np.asarray(m.weights_);yield np.asarray(m._weight_sum);yield np.asarray(m.classes_)
def fitted_digest(m):
    h=hashlib.sha256()
    for a in scientific_arrays(m):
        if a.dtype.kind in 'fc':assert np.isfinite(a).all(),'Nonfinite native fitted state cannot become an accuracy negative'
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
    metadata=(m.n_kernels,m.n_estimators,m.n_estimators_,m.n_cases_,m.n_channels_,m.n_timepoints_,m.rocket_transform,m.class_weight,m.n_jobs,m.random_state,[(p.steps[0][1].random_state,p.steps[0][1].normalise,int(p.steps[1][1].n_samples_seen_)) for p in m.estimators_]);h.update(repr(metadata).encode());return h.hexdigest()
def predict(m,maps,locations,ids):
    before=fitted_digest(m);probas=[];hashes=[];member_predictions=[]
    for start in range(0,len(locations),64):
        loc=locations[start:start+64];x=np.ascontiguousarray(maps[loc].transpose(0,2,1));z=Normalizer().fit_transform(x).astype(np.float32,copy=False)
        probability=np.zeros((len(loc),12));batch=hashlib.sha256();votes=[]
        for pipe,w in zip(m.estimators_,m.weights_):
            t,s,r=[v for _,v in pipe.steps];raw=t._transform_kernels(z);assert raw.shape==(len(loc),4000) and np.isfinite(raw).all()
            f=s.transform(raw);assert np.isfinite(f).all();batch.update(np.ascontiguousarray(f).tobytes());p=r.predict(f);votes.append(p);probability[np.arange(len(loc)),np.searchsorted(m.classes_,p)]+=w
        probability/=m._weight_sum;assert np.isfinite(probability).all();np.testing.assert_allclose(probability.sum(1),1,rtol=1e-12,atol=1e-12)
        probas.append(probability);hashes.append(batch.hexdigest());member_predictions.append(np.stack(votes,axis=1))
    p=np.concatenate(probas);assert fitted_digest(m)==before
    return dict(ids=ids[locations],probabilities=p,predictions=canonical_predictions(m,p),member_predictions=np.concatenate(member_predictions),feature_batch_digests=np.asarray(hashes))
