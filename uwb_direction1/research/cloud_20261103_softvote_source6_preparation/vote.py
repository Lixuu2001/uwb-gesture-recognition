"""Fixed T1 links and uniform probability mixture. No labels, fit or tuning."""
import numpy as np
def probabilities(logits,link):
    z=np.asarray(logits,dtype=np.float64)
    assert z.ndim==2 and z.shape[1]==12 and len(z)>0 and np.isfinite(z).all()
    if link=='ovr_bce':z=-np.logaddexp(0.,-z)
    else:assert link=='categorical_ce'
    e=np.exp(z-z.max(axis=1,keepdims=True));p=e/e.sum(axis=1,keepdims=True)
    assert np.isfinite(p).all() and np.all(p>=0)
    return p
def combine(ids_a,logits_a,ids_b,logits_b):
    a=np.asarray(ids_a);b=np.asarray(ids_b)
    assert a.ndim==b.ndim==1 and np.issubdtype(a.dtype,np.integer) and np.issubdtype(b.dtype,np.integer)
    assert len(a)==len(np.unique(a)) and len(b)==len(np.unique(b))
    assert np.array_equal(a,b),'Sample-ID order must match exactly; no implicit reorder'
    assert len(a)==len(logits_a)==len(logits_b)
    pa=probabilities(logits_a,'ovr_bce');pb=probabilities(logits_b,'categorical_ce')
    q=.5*pa+.5*pb
    return q,q.argmax(1)
