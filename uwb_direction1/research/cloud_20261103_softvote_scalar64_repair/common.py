from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def write(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    assert not t.exists()
    t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def ready():
    assert not sys.flags.optimize
    m=read(B/'REMOTE_PREREGISTRATION.json')
    assert m['state']=='all_changed_files_exact_remote_readback_verified'
    assert m['branch_head_verified'] is True and m['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
    for n,h in read(B/'INPUT_HASHES.json').items():assert sha(n)==h,n
    for n,h in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
    p=read(B/'PROTOCOL.json')
    old=Path(p['operational_repair']['original_root']);original=read(old/'PROTOCOL.json')
    assert all(p[k]==v for k,v in original.items()),'Scientific protocol must remain identical'
    assert not (old/'audit01/DIAGNOSTIC_UNSCORED.npz').exists()
    assert not (old/'FINALIZE_ONCE.lock').exists()
    assert sha(B/'validation01/OUTPUTS.npz')==sha(old/'validation01/OUTPUTS.npz')
    assert sha(B/'validation01/RESULT.json')==sha(old/'validation01/RESULT.json')
    assert p['weights']==[.5,.5] and p['temperature']==1 and p['seed']==20261029
    assert p['minimum_validation_gain_over_best_member']==.001
    assert p['minimum_diagnostic_gain']==.005 and p['reference_macro']==.9866666666666667
    assert p['users7_8_labels_forbidden'] and p['new_training'] is False
    return p
def ids_maps(p):
    import numpy as np
    with np.load(p['ids']) as z:ids=z['ids'];users=z['users']
    assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900
    assert set(np.unique(users))==set(range(1,7))
    val=np.flatnonzero(users==2);held=np.flatnonzero(users==1)
    assert len(val)==1000 and len(held)==1200
    maps=np.load(p['maps'],mmap_mode='r')
    assert maps.shape==(6900,90,567) and maps.dtype==np.float32
    return ids,users,val,held,maps
def macro(y,pred):
    import numpy as np
    assert y.shape==pred.shape
    return float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)]))
def validation_pass(scores,p):
    return scores['mixture']+p['tolerance']>=max(scores['timemil'],scores['shuffle'])+p['minimum_validation_gain_over_best_member']
def complete():
    p=ready()
    for phase in ['validate','audit']:
        t=read(B/(phase+'_supervisor01')/'TERMINATION.json')
        assert t['exit_code']==0 and t['resource_stop_reason'] is None and not t['wall_timeout']
        assert t['cgroup_events_available'] and t['cgroup_oom_kill_delta']==0
    v=read(B/'validation01/RESULT.json');a=read(B/'audit01/CERTIFICATE.json')
    assert a['state']=='pass' and a['diagnostic_labels_read'] is False and a['users7_8_labels_used'] is False
    assert a['validation_fresh_TimeMIL_replay_exact'] and a['all1000_probabilities_independent_scalar_verified']
    assert a['validation_gate_passed']==v['validation_gate_passed']==validation_pass(v['scores'],p)
    assert a['all1200_unscored_diagnostic_probabilities_independent_scalar_verified']==v['validation_gate_passed']
    for n,h in a['artifact_hashes'].items():assert sha(n)==h,n
    assert v['outputs_sha256']==sha(B/'validation01/OUTPUTS.npz')
    assert (B/'audit01/DIAGNOSTIC_UNSCORED.npz').exists()==v['validation_gate_passed']
    return p,v,a
