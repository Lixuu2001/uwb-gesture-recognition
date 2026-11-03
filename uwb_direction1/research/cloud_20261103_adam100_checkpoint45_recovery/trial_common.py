"""UNEXECUTED real-trial I/O; no supplied activation files authorize execution."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    assert not tmp.exists()
    with tmp.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    tmp.replace(path)
def ready():
    if sys.flags.optimize:raise RuntimeError('Assertions must remain enabled')
    m=read(B/'REMOTE_REAL_PREREGISTRATION.json')
    assert m['state']=='real_trial_exact_remote_readback_verified' and m['branch_head_verified'] is True
    assert m['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    frozen=read(B/'FROZEN_HASHES.json')
    for name,digest in frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(B/q)==digest,name
    for name,digest in read(B/'DEPENDENCIES.json')['files'].items():assert sha(name)==digest,name
    c=read(B/'TRIAL_PROTOCOL.json');old=Path(c['operational_recovery']['original_root'])
    import importlib.util
    spec=importlib.util.spec_from_file_location('original_checkpoint45_common',old/'trial_common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.ready()
    for key,value in original.items():
        if key!='archive_upper_bytes':assert c[key]==value,key
    assert c['epochs']==100 and c['batch_size']==20 and c['seed']==20261029
    recovery=read(B/'RECOVERY_INPUTS.json')
    assert recovery['checkpoint_epoch']==45 and recovery['checkpoint_updates']==10575
    assert recovery['checkpoint_sha256']==sha(old/'run01/final_full.pt')
    for name,digest in recovery['files'].items():assert sha(name)==digest,name
    tests=read(B/'RECOVERY_SYNTHETIC_CHECK.json')
    assert tests['state']=='pass' and tests['real_UWB_fit_started'] is False
    assert tests['native_full_B20_resume_permutation_next_updates_exact'] is True
    assert tests['uncommitted_suffix_refusal_and_prefix_copy_boundaries'] is True
    sys.path.insert(0,c['source_module_directory'])
    return c
def load_inputs(c, labels=True):
    from native_adam100 import np
    assert sha(c['source_manifest'])==c['source_manifest_sha256']
    assert sha(c['source_maps'])==c['source_maps_sha256']
    for n,h in c['input_sha256'].items():assert sha(Path(c['data'])/'cache'/n)==h,n
    with np.load(c['source_manifest']) as z:
        ids=z['ids'];users=z['users']
    assert ids.shape==users.shape==(6900,)
    assert np.issubdtype(ids.dtype,np.integer) and len(np.unique(ids))==6900
    assert int(ids.min())>=0 and int(ids.max())<9593
    assert set(np.unique(users).tolist())==set(range(1,7))
    assert [(users==u).sum() for u in range(1,7)]==[1200,1000,1200,1200,1200,1100]
    loc={k:np.flatnonzero(test) for k,test in [
        ('train',users>=3),('validation',users==2),('diagnostic',users==1)]}
    assert {k:len(v) for k,v in loc.items()}==dict(train=4700,validation=1000,diagnostic=1200)
    maps=np.load(c['source_maps'],mmap_mode='r')
    assert maps.dtype==np.float32 and maps.shape==(6900,90,567)
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r')
    np.testing.assert_array_equal(subj[ids],users)
    y=None;v=None
    if labels:
        truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
        # Decode only the approved source train and validation rows.
        y=np.array(truth[ids[loc['train']]],copy=True)
        v=np.array(truth[ids[loc['validation']]],copy=True)
        assert set(np.unique(y).tolist())==set(range(12))
        assert set(np.unique(v).tolist())==set(range(10))
    return ids,users,loc,maps,y,v

def term_ok(t):
    return (type(t.get('exit_code')) is int and t['exit_code']==0
        and t.get('wall_timeout') is False
        and 'resource_stop_reason' in t and t['resource_stop_reason'] is None
        and type(t.get('cgroup_oom_kill_delta')) is int
        and t['cgroup_oom_kill_delta']==0
        and t.get('cgroup_events_available') is True)

def complete():
    c=ready()
    for name in ['supervisor01','audit_supervisor01']:
        assert term_ok(read(B/name/'TERMINATION.json'))
    a=read(B/'COMPLETE_AUDIT.json')
    assert a['state']=='pass' and a['diagnostic_labels_read'] is False
    assert a['users7_8_labels_used'] is False
    for key in ['full100_epoch23500_update_replay_exact',
            'all100_validation_outputs_exact','fixed_final100_endpoint_exact',
            'full_model_BN_Adam_two_moments_steps_order_and_Torch_RNG_exact',
            'all2200_final_scores_predictions_exact',
            'all6900_raw_RMS_cache_FP64_verified']:
        assert a[key] is True,key
    assert type(a['pixel_count']) is int and a['pixel_count']==1038643200
    required=['initial12_head.pt','final_model.pt','final_full.pt','split.npz',
        'steps.jsonl','TRAIN_ENDPOINT.json','FILES.json','IDENTITY.json',
        'STATE.json','validation_unscored.npz','diagnostic_unscored.npz']
    required += ['validation_epoch%03d.npz'%i for i in range(1,101)]
    required += ['epoch%03d.json'%i for i in range(1,101)]
    hashes=a['artifact_hashes']
    assert all(str(B/'run01'/name) in hashes for name in required)
    for name,digest in hashes.items():
        assert Path(name).resolve().is_relative_to(B.resolve())
        assert sha(name)==digest,name
    return c
