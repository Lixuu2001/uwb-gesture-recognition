"""Explicit source-only roles; diagnostic labels deliberately have no loader."""
def split(source_ids,users,np,counts):
 if source_ids.dtype.kind not in 'iu' or users.dtype.kind not in 'iu' or source_ids.shape!=users.shape or len(source_ids)!=6900 or len(np.unique(source_ids))!=6900 or np.any(source_ids<0) or set(users.tolist())!=set(range(1,7)):raise ValueError('Wrong source identity/user scope')
 ix={'train':np.flatnonzero(np.isin(users,[3,4,5,6])),'validation':np.flatnonzero(users==2),'diagnostic':np.flatnonzero(users==1)}
 if {k:len(v) for k,v in ix.items()}!=counts:raise ValueError('Wrong fixed counts')
 sets=[set(v.tolist()) for v in ix.values()]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or set.union(*sets)!=set(range(6900)):raise ValueError('Role leakage/incomplete coverage')
 return ix
def evaluation_order(count,role,seed,np):
 if role not in ['validation','diagnostic']:raise ValueError('Unknown inference role')
 return np.random.RandomState(seed+(101 if role=='validation' else 102)).permutation(count)
def load(c):
 from engine import np
 from pathlib import Path
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 ix=split(ids,users,np,c['counts']);ordered={role:ix[role][evaluation_order(len(ix[role]),role,c['seed'],np)] for role in ['validation','diagnostic']}
 maps=np.load(c['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
 subjects=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');assert np.all(ids<len(subjects));np.testing.assert_array_equal(subjects[ids],users)
 labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
 # Only training and validation indices are ever used here.
 yt=np.array(labels[ids[ix['train']]],copy=True);yv=np.array(labels[ids[ordered['validation']]],copy=True)
 x=np.array(maps[ix['train']].transpose(0,2,1),dtype='float32',copy=True,order='C');vx=np.array(maps[ordered['validation']].transpose(0,2,1),dtype='float32',copy=True,order='C')
 return ids,users,ix,ordered,maps,x,yt,vx,yv
