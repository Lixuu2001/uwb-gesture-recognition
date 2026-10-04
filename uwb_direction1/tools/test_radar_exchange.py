import numpy as np
from radar_exchange import Q,S,arms
rng=np.random.default_rng(7);a=rng.normal(size=(23,12));b=rng.normal(size=(23,12));np.testing.assert_array_equal(Q[Q],np.arange(12));np.testing.assert_array_equal(np.array(S)[S],np.arange(3))
p=arms(a,b);r=arms(b,a);np.testing.assert_array_equal(r['equivariant'],p['equivariant'][:,Q]);np.testing.assert_array_equal(r['invariant'],p['invariant'])
for i,j in [(0,1),(4,6),(5,7),(8,9)]:assert Q[i]==j and Q[j]==i
for i in [2,3,10,11]:assert Q[i]==i
# Exact equivariance alone is vacuous about accuracy: arbitrary scores obey it too.
print('passed: semantic involution, slot involution, random-score projection equivariance, invariant control')
