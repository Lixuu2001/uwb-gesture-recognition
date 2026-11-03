from common import B,read,write,sha,complete,ids_maps,macro
from backup import privacy
from stream_archive import verify
import os
def main():
    p,v,a=complete();b=B/'private_backup';m=read(b/'MANIFEST.json');rows=read(b/'READBACK_INPUT.json');privacy(rows)
    r=read(b/'DRIVE_BACKUP_RECEIPT.json');verified=read(b/'READBACK_VERIFIED.json')
    assert r['state']=='private_raw_all_members_verified' and r['complete_selected_modelset_verified'] is True
    assert r['diagnostic_labels_scored'] is False and r['validation_gate_passed']==v['validation_gate_passed']
    assert r['scope']==m['scope']==verified['scope']==p['private_scope']
    assert r['archive_sha256']==m['archive_sha256']==verified['archive_sha256']
    assert sha(b/'MEMBERS.json')==r['members_manifest_sha256']==m['members_manifest_sha256']
    assert verify(m,rows)==verified
    entries=read(b/'MEMBERS.json')['members'];members={x['path']:x for x in entries}
    targets={'validation01/OUTPUTS.npz':B/'validation01/OUTPUTS.npz','audit01/CERTIFICATE.json':B/'audit01/CERTIFICATE.json',
        'selected_models/TimeMIL/best.pt':__import__('pathlib').Path(p['timemil_root'])/'run02/timemil/best.pt',
        'selected_models/ShuffleNet/final_model.pt':__import__('pathlib').Path(p['shuffle_root'])/'run01/final_model.pt'}
    if v['validation_gate_passed']:targets['audit01/DIAGNOSTIC_UNSCORED.npz']=B/'audit01/DIAGNOSTIC_UNSCORED.npz'
    for n,f in targets.items():assert members[n]['sha256']==sha(f)
    assert not (B/'SUMMARY.json').exists()
    fd=os.open(B/'FINALIZE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    score=None;delta=None;passed=False
    if v['validation_gate_passed']:
        import numpy as np
        ids,users,va,held,maps=ids_maps(p)
        with np.load(B/'audit01/DIAGNOSTIC_UNSCORED.npz') as z:
            np.testing.assert_array_equal(z['ids'],ids[held]);pred=z['predictions']
            np.testing.assert_array_equal(pred,z['probabilities'].argmax(1))
        truth=np.load(p['data']+'/cache/y_clutterremoved.npy',mmap_mode='r');y=np.array(truth[ids[held]],copy=True)
        assert set(np.unique(y))==set(range(12));score=macro(y,pred);delta=score-p['reference_macro'];passed=delta+p['tolerance']>=p['minimum_diagnostic_gain']
    result=dict(state='complete_exploratory_uniform_probability_mixture',validation=v,
        diagnostic_macro=score,delta_over_reference=delta,predeclared_diagnostic_gate_passed=passed,
        diagnostic_labels_read=v['validation_gate_passed'],users7_8_labels_used=False,
        route='separately_register_source6_confirmation' if passed else p['fail_route'],
        complete_private_selected_models_and_outputs_verified=True,independent_validation=False,novel_contribution_validated=False)
    write(B/'SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
