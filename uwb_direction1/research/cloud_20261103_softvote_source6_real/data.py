"""Metadata role checks precede selective training/validation label decoding."""
from pathlib import Path
from common import B,config
from roles_gate import roles
def load(held,labels=False):
    import numpy as np
    c,p=config()
    with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    loc=roles(ids,users,held)
    with np.load(B/'assets'/f'held{held}_roles.npz') as z:
        for role,positions in loc.items():np.testing.assert_array_equal(z[role],ids[positions])
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r')
    np.testing.assert_array_equal(subj[ids],users)
    maps=np.load(c['maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
    y=v=None
    if labels:
        truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
        y=np.array(truth[ids[loc['train']]],copy=True)
        v=np.array(truth[ids[loc['validation']]],copy=True)
        assert set(np.unique(y))==set(range(12)) and len(v)==len(loc['validation'])
    return ids,users,loc,maps,y,v
def macro(y,pred):
    import numpy as np
    return float(np.mean([np.mean(pred[y==k]==k) for k in np.unique(y)]))
