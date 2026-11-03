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
    if sys.flags.optimize:raise RuntimeError('Do not disable assertions')
    c=read(B/'TRIAL_PROTOCOL.json')
    assert c['state']=='selected_frozen_real_trial'
    m=read(B/'REMOTE_REAL_PREREGISTRATION.json')
    assert m['state']=='real_trial_exact_remote_readback_verified'
    assert m['branch_head_verified'] is True
    assert m['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    frozen=read(B/'FROZEN_HASHES.json')
    required=['TRIAL_PROTOCOL.json','trial_common.py','adam_engine.py',
        'produce.py','audit_full.py','audit_pixels.py','DEPENDENCIES.json',
        'SOURCE_TESTS.log','independent_supervisor.py','controller.py',
        'stream_archive.py','backup.py','finalize_once.py',
        'test_training_integrity.py','test_completion_boundary.py',
        'test_stream_archive.py','STREAM_TESTS.log','COMPLETION_TESTS.log',
        'TRAINING_TESTS.log','PIPELINE_SYNTHETIC_CHECK.json',
        'build_initial_insurance.py','test_supervisor.py','test_controller.py',
        'test_pipeline_controls.py','SUPERVISOR_TESTS.log','CONTROLLER_PIPELINE_TESTS.log']
    assert all(n in frozen for n in required)
    for name,h in frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(B/q)==h,name
    pipeline=read(B/'PIPELINE_SYNTHETIC_CHECK.json')
    assert pipeline['state']=='all_registered_trial_boundaries_passed'
    for key in ['full_Adam_checkpoint_next_updates_exact',
        'controller_refuses_duplicate_or_unknown_prior_trial',
        'producer_and_audit_resource_termination_recorded',
        'private_stream_all_members_local_recovery_exact_raw_Drive_required_before_controller',
        'finalizer_requires_full100_audit_private_readback_once_lock']:
        assert pipeline[key] is True
    deps=read(B/'DEPENDENCIES.json')['files']
    native=(B/c['source_module_directory']).resolve()
    assert str(native/'native_adam100.py') in deps
    for name,h in deps.items():assert sha(name)==h,name
    assert (c['epochs'],c['batch_size'],c['updates'],c['seed'],c['threads'])==(
        100,20,23500,20261029,4)
    assert (c['lr'],c['betas'],c['eps'],c['weight_decay'],c['augmentation'],
        c['checkpoint_selection'])==(.001,[.9,.999],1e-8,0,False,'fixed_epoch100')
    assert c['users7_8_labels_forbidden'] is True
    cost=Path(c['cost_workspace']).resolve()
    assert str(cost/'VERIFIED_COST_RESULT.json') in deps
    assert str(cost/'cost_guard.py') in deps
    sys.path.insert(0,str(cost))
    from cost_guard import ready as cost_ready
    cost_policy=cost_ready()
    assert cost_policy['epochs']==100 and cost_policy['batch_size']==20
    r=read(cost/'VERIFIED_COST_RESULT.json')
    assert r['state']=='verified_full_ShuffleNetV2_x1_Adam100_B20_synthetic_cost'
    assert r['resource_gate_passed'] is True and r['real_training_started'] is False
    assert m['verified_cost_sha256']==sha(cost/'VERIFIED_COST_RESULT.json')
    sys.path.insert(0,str(native))
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
