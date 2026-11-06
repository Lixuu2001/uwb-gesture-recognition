"""Per-fold subject firewall; held labels are never sliced in production workers."""
from pathlib import Path
import numpy as np
from common import protocol
from loso_roles_metrics import roles
def load(labels=False):
    c=protocol()
    with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900
    val,loc=roles(users,c['held']);assert val==c['validation_user']
    assert list(map(len,loc.values()))==c['counts']
    assert set(users[loc['train']])==set(c['train_users'])
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
    maps=np.load(c['maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
    y=v=None
    if labels:
        truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
        y=np.array(truth[ids[loc['train']]],copy=True);v=np.array(truth[ids[loc['validation']]],copy=True)
        assert set(np.unique(y))==set(range(12))
    return ids,users,loc,maps,y,v
