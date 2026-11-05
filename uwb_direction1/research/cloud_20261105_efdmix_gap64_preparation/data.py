"""Only source metadata and train/validation labels before diagnostic freeze."""
from common import protocol
import numpy as np
from pathlib import Path
def load(labels=False):
    c=protocol()
    with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900 and set(users)==set(range(1,7))
    assert [(users==u).sum() for u in range(1,7)]==[1200,1000,1200,1200,1200,1100]
    loc=dict(train=np.flatnonzero(np.isin(users,c['train_users'])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
    assert tuple(map(len,loc.values()))==(4700,1000,1200)
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
    maps=np.load(c['maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
    y=v=None
    if labels:
        truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
        y=np.array(truth[ids[loc['train']]],copy=True);v=np.array(truth[ids[loc['validation']]],copy=True)
        assert set(np.unique(y))==set(range(12))
    return ids,users,loc,maps,y,v
