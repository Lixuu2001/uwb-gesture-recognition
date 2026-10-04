import numpy as np
from scipy.linalg import solve
from scipy import sparse
from query_graph import probabilities,graphs,smooth

rng=np.random.default_rng(6);x=rng.normal(size=(21,5));x[1]=x[0];x[2]=0;ids=rng.permutation(np.arange(21));p=probabilities(rng.normal(size=(21,12)));gg=graphs(x,ids,k=4)
for name,w in gg.items():
 f,_=smooth(p,w);a=w.toarray();np.testing.assert_allclose(f,solve(np.eye(21)+np.diag(a.sum(1))-a,p,assume_a='pos'),atol=1e-12)
 order=rng.permutation(21);ww=graphs(x[order],ids[order],k=4)[name];np.testing.assert_array_equal(ww.toarray(),a[order][:,order]);ff,_=smooth(p[order],ww);np.testing.assert_allclose(ff,f[order],atol=1e-12)
 np.testing.assert_allclose(f.sum(1),1,atol=1e-12);assert f.min()>=0
f,_=smooth(p,sparse.csr_matrix((21,21)));np.testing.assert_array_equal(f,p)
assert graphs(x[:1],ids[:1])['mutual'].nnz==0
print('PASS dense/sparse solve, simplex, duplicate-distance stable-ID ties, input permutation equivariance including graph null, isolated identity')
