import sys,hashlib
from common import read,B
sys.path.insert(0,'/workspace/uwb-minirocket-native-source-probe-20261102')
from native_minirocket import np,make,assert_native
def fitted_digest(m):
    h=hashlib.sha256()
    for a in list(m._transformer.parameters)+[m._scaler.mean_,m._scaler.var_,m._scaler.scale_,m.estimator_.coef_,m.estimator_.intercept_,m.estimator_.classes_]:
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(float(m.estimator_.alpha_)).encode());h.update(repr(float(m.estimator_.best_score_)).encode())
    return h.hexdigest()
def predict(m,maps,locations,ids):
    before=fitted_digest(m);dec=[];pred=[];digests=[]
    for start in range(0,len(locations),64):
        loc=locations[start:start+64];x=np.ascontiguousarray(maps[loc].transpose(0,2,1))
        f=m.pipeline_[:-1].transform(x);z=m.estimator_.decision_function(f)
        p=m.estimator_.predict(f);np.testing.assert_array_equal(p,m.estimator_.classes_[z.argmax(1)])
        assert np.isfinite(f).all() and np.isfinite(z).all()
        digests.append(hashlib.sha256(f.tobytes()).hexdigest());dec.append(z);pred.append(p)
    assert fitted_digest(m)==before
    return dict(ids=ids[locations],decisions=np.concatenate(dec),predictions=np.concatenate(pred),feature_batch_digests=np.array(digests))
