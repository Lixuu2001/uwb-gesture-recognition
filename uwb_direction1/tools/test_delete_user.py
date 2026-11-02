import numpy as np
from delete_user import subsets,aggregate
rng=np.random.default_rng(6);y=np.tile(np.arange(12).repeat(10),4);u=np.repeat(np.arange(3,7),120);keep=~((u==6)&(y==3));y=y[keep];u=u[keep];ids=np.arange(1000,1000+len(y));m=subsets(ids,y,u,8,20261007);lookup={v:i for i,v in enumerate(ids)}
for d in range(3,7):
    a=m[f'domain_{d}'];b=m[f'row_{d}'];assert len(a)==len(b) and len(set(b))==len(b)
    assert d not in u[[lookup[v] for v in a]]
    np.testing.assert_array_equal(np.bincount(y[[lookup[v] for v in a]],minlength=12),np.bincount(y[[lookup[v] for v in b]],minlength=12))
scores={k:rng.normal(size=(20,12)) for k in m};a=aggregate(scores);conv=rng.dirichlet(np.ones(4));stack=np.array([scores[f'domain_{d}'] for d in range(3,7)]);assert np.all(np.einsum('i,ijk->jk',conv,stack)>=a['domain_min']-1e-12)
base=rng.normal(size=(20,12));a=aggregate({k:base for k in m});np.testing.assert_array_equal(a['domain_min'],base);np.testing.assert_array_equal(a['row_mean'],base)
print('passed: missing-class count matching, no replacement, source deletion, convex-mixture minimum bound and identical-model reduction')
