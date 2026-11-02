import sys,hashlib
sys.path.insert(0,'/workspace/uwb-hydra-multirocket-source-probe-20261102')
from native_hydramr import np,torch,make,assert_native,components
def scientific_arrays(m):
    h=m._transform_hydra._hydra
    tensors=[h.W,h.dilations,h.paddings,*h.idx,m._scale_hydra.mu,m._scale_hydra.sigma,m._scale_hydra.epsilon]
    arrays=[t.detach().cpu().numpy() for t in tensors]
    arrays+=list(m._transform_multirocket.parameter)+list(m._transform_multirocket.parameter1)
    arrays += [m._scale_multirocket.mean_,m._scale_multirocket.var_,m._scale_multirocket.scale_,m.classifier.coef_,m.classifier.intercept_,m.classifier.classes_]
    return arrays
def fitted_digest(m):
    digest=hashlib.sha256()
    for a in scientific_arrays(m):
        a=np.asarray(a);digest.update(str(a.shape).encode());digest.update(str(a.dtype).encode());digest.update(np.ascontiguousarray(a).tobytes())
    metadata=[m.n_groups,m.n_kernels,m.random_state,m.class_weight,m.n_jobs,m._scale_hydra.mask,m._scale_hydra.exponent,int(m._scale_multirocket.n_samples_seen_),float(m.classifier.alpha_),float(m.classifier.best_score_)]
    digest.update(repr(metadata).encode());return digest.hexdigest()
def predict(m,maps,locations,ids):
    before=fitted_digest(m);ds=[];ps=[];digests=[]
    for start in range(0,len(locations),64):
        loc=locations[start:start+64];x=np.ascontiguousarray(maps[loc].transpose(0,2,1))
        h,raw,f=components(m,x);assert np.isfinite(h.numpy()).all() and np.isfinite(f).all()
        z=m.classifier.decision_function(f);p=m.classifier.predict(f)
        np.testing.assert_array_equal(p,m.classifier.classes_[z.argmax(1)]);assert np.isfinite(z).all()
        ds.append(z);ps.append(p);digests.append(hashlib.sha256(f.tobytes()).hexdigest())
    assert fitted_digest(m)==before
    return dict(ids=ids[locations],decisions=np.concatenate(ds),predictions=np.concatenate(ps),feature_batch_digests=np.array(digests))
