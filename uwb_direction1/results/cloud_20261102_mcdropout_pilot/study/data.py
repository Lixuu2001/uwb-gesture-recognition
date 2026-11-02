"""No gesture label loader exists for MC inference or replay."""
def load(c):
 from pathlib import Path
 from engine import np
 with np.load(c['source_manifest']) as a:ids=a['ids'];users=a['users']
 assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900 and set(users.tolist())==set(range(1,7))
 ordered={role:np.flatnonzero(users==u) for role,u in [('validation',2),('diagnostic',1)]};assert {k:len(v) for k,v in ordered.items()}==dict(validation=1000,diagnostic=1200)
 maps=np.load(c['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32;subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
 return ids,users,ordered,maps
