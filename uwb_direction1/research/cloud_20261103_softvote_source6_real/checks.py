from pathlib import Path
from common import B,read,sha
def member(held,arm):return B/'folds'/f'held{held}'/arm
def verify_member(held,arm):
    out=member(held,arm);r=read(out/'ARM_COMPLETE.json')
    assert r['state']=='complete_member_unscored' and r['held']==held and r['arm']==arm
    assert r['epochs']==(64 if arm=='timemil' else 100)
    assert not r['diagnostic_labels_read'] and not r['users7_8_labels_used']
    for n,h in r['files'].items():assert sha(out/n)==h,n
    return out,r
def verify_audits(held):
    for arm in ['timemil','shuffle']:
        out,r=verify_member(held,arm)
        first=read(out/'first_audit/CERTIFICATE.json');assert first['state']=='pass'
        for phase in ['first_audit','inference_audit']:
            t=read(out/phase/'TERMINATION.json')
            assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
        cert=read(out/'inference_audit/CERTIFICATE.json')
        assert cert['state']=='pass' and not cert['diagnostic_labels_read'] and not cert['users7_8_labels_used']
        for n,h in cert['artifact_hashes'].items():assert sha(n)==h
        if arm=='shuffle':
            t=read(out/'replay/TERMINATION.json')
            assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
            full=read(out/'replay/COMPLETE_AUDIT.json')
            assert full['state']=='pass' and full['all100epochs_all_updates_replayed_exactly']
            assert full['initial_audit_started_from_fresh_seed']
            assert full['replay_progress_sha256']==sha(out/'replay/PROGRESS.json')
    return True
