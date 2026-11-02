from pathlib import Path
import hashlib,json,os,pickle,time
B=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def save(p,v):
    with Path(p).open('wb') as f:pickle.dump(v,f,protocol=5)
def ready(marker=True):
    if marker:
        r=read(B/'REMOTE_PREREGISTRATION.json')
        assert r['state']=='real_pilot_remote_exact_readback_verified' and r['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    for p,d in read(B/'FROZEN_HASHES.json').items():assert sha(B/p)==d,p
    for p,d in read(B/'DEPENDENCIES.json')['files'].items():assert sha(p)==d,p
    c=read(B/'PROTOCOL.json');cost=read(c['cost_workspace']+'/VERIFIED_COST_RESULT.json')
    assert cost['state']=='independently_verified_full4700_official_ResNet18_frozen512_LinearSVC_C1_cost' and cost['resource_gate_passed'] and not cost['real_UWB_values_labels_fit']
    return c
def load_inputs(c,training_labels=False):
    import sys
    sys.path.insert(0,c['source_preparation'])
    from native_resnet18 import np
    with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
    assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900
    assert set(users.tolist())==set(range(1,7))
    loc={'train':np.flatnonzero(users>=3),'validation':np.flatnonzero(users==2),'diagnostic':np.flatnonzero(users==1)}
    assert {k:len(v) for k,v in loc.items()}==dict(train=4700,validation=1000,diagnostic=1200)
    maps=np.load(c['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r')
    np.testing.assert_array_equal(subj[ids],users)
    labels=None
    if training_labels:
        assert set(users[loc['train']].tolist())=={3,4,5,6}
        y=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
        labels=np.array(y[ids[loc['train']]],copy=True);assert set(labels.tolist())==set(range(12))
    return ids,users,loc,maps,labels
def complete():
    c=ready();t=read(B/'supervisor01/TERMINATION.json');a=read(B/'audit_supervisor01/TERMINATION.json')
    assert t['exit_code']==a['exit_code']==0 and not t['wall_timeout'] and not a['wall_timeout'] and not t['resource_stop_reason'] and not a['resource_stop_reason']
    r=read(B/'COMPLETE_AUDIT.json');assert r['state']=='pass' and not r['diagnostic_labels_read'] and not r['users7_8_labels_used']
    for p,d in r['artifact_hashes'].items():assert sha(p)==d,p
    return c
