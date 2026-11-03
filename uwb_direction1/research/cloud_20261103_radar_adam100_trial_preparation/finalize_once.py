"""Prospective one-time held1 gate. No scoring before complete/private evidence."""
from pathlib import Path
import os,json
from trial_common import B, complete, read, write, sha, load_inputs
from stream_archive import verify
from backup import SCOPE, privacy

def require_private(c):
    p=B/'complete_backup';m=read(p/'MANIFEST.json')
    v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json')
    assert m['scope']==v['scope']==r['scope']==SCOPE
    assert m['complete_modelset'] is True
    assert r['state']=='private_readback_verified'
    assert r['complete_modelset_verified'] is True and r['numeric_audit_certified'] is True
    assert r['diagnostic_labels_scored'] is False
    assert v['state']=='remote_archive_and_all_members_verified'
    assert v['complete_modelset_verified'] is True
    assert m['archive_sha256']==v['archive_sha256']==r['archive_sha256']
    assert sha(p/'MEMBERS.json')==m['members_manifest_sha256']==r['members_manifest_sha256']
    rows=read(p/'READBACK_INPUT.json');privacy(rows);actual=verify(m,rows)
    assert actual==v
    members=read(p/'MEMBERS.json')['members'];bound={x['path']:x for x in members}
    assert len(bound)==len(members)==m['tracked_payload_files']
    targets={'run01/'+n:B/'run01'/n for n in ['initial12_head.pt',
        'final_model.pt','final_full.pt','split.npz','steps.jsonl',
        'validation_unscored.npz','diagnostic_unscored.npz']}
    targets['study/COMPLETE_AUDIT.json']=B/'COMPLETE_AUDIT.json'
    targets['dependencies/source_ids.npz']=Path(c['source_manifest'])
    for name,path in targets.items():
        assert bound[name]['sha256']==sha(path) and bound[name]['bytes']==path.stat().st_size
    assert bound['official/weights/shufflenetv2_x1-5666bf0f80.pth']['sha256']==c['weights_sha256']

def main():
    c=complete();require_private(c)
    assert not (B/'run01/SUMMARY.json').exists()
    fd=os.open(B/'FINALIZE_ONCE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    from adam_engine import np
    ids,users,loc,maps,_,_=load_inputs(c,labels=False)
    truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={}
    for role,support in [('validation',range(10)),('diagnostic',range(12))]:
        with np.load(B/'run01'/(role+'_unscored.npz')) as z:
            np.testing.assert_array_equal(z['ids'],ids[loc[role]])
            assert z['scores'].shape==(len(loc[role]),12) and np.isfinite(z['scores']).all()
            np.testing.assert_array_equal(z['predictions'],z['scores'].argmax(1))
            y=np.array(truth[z['ids']],copy=True);pred=z['predictions']
            assert set(np.unique(y).tolist())==set(support)
            scores[role]=float(np.mean([np.mean(pred[y==k]==k) for k in support]))
    delta=scores['diagnostic']-c['reference_macro'];passed=delta+c['tolerance']>=c['minimum_gain']
    result=dict(state='complete_exploratory_full_ShuffleNetV2_x1_Adam100_B20',
        scores=scores,delta_macro=delta,predeclared_gate_passed=passed,
        route='separately_register_source6_confirmation' if passed else c['fail_route'],
        complete_private_modelset_verified=True,source6_confirmation_complete=False,
        users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False)
    write(B/'run01/SUMMARY.json',result);print(json.dumps(result))

if __name__=='__main__':main()
