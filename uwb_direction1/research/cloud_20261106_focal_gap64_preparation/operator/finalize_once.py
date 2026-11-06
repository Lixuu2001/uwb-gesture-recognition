"""Locked source1 accuracy gate only after full private raw verification."""
from pathlib import Path
import sys,os,json
sys.path.insert(0,'/workspace/uwb-focal-gap64-20261106')
from common import B,read,write,sha,ready
from stream_archive import verify
def main():
    c=ready(True);root=B/'run';release=read(B/'REAL_RELEASE.json')
    assert sha(__file__)==release['once_scorer_sha256']
    assert not (B/'SUMMARY.json').exists() and not (B/'FINALIZE_ONCE.lock').exists()
    q=read(B/'CONTROLLER_IDENTITY.json')
    from controller import identity
    now=identity(q['pid']);assert now is None or now['state']=='Z' or now['startticks']!=q['startticks'] or now['boot_id']!=q['boot_id']
    assert read(B/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
    for stage in ['train','replay','output']:
        t=read(root/'execution'/stage/'TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None
    frozen=read(root/'PREDICTIONS_FROZEN.json');assert frozen['full_training_replayed'] and not frozen['diagnostic_labels_read']
    for n,h in frozen['files'].items():assert sha(root/n)==h
    d=B/'full_backup';m=read(d/'MANIFEST.json');rows=read(d/'READBACK_INPUT.json');v=read(d/'READBACK_VERIFIED.json')
    for r in rows:
        z=r['metadata'];assert not z['shared'] and z['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and z['permissions']==[dict(role='owner',type='user')]
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    assert verify(m,rows)==v and m['scope']=='fixed_Focal_GAP64_full_pilot_models_audits_outputs_before_once_gate'
    assert sha(d/'MEMBERS.json')==m['members_manifest_sha256']
    members={x['path']:x for x in read(d/'MEMBERS.json')['members']}
    for n in ['train/selected.pt','train/final_full.pt','train/diagnostic_unscored.npz','replay/COMPLETE.json','PREDICTIONS_FROZEN.json']:
        assert members['run/'+n]['sha256']==sha(root/n)
    # Lock before the first new diagnostic truth slice; never rescore on recovery.
    fd=os.open(B/'FINALIZE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    from data import load
    import numpy as np
    ids,u,loc,maps,_,_=load(False)
    with np.load(root/'train/diagnostic_unscored.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids[loc['diagnostic']]);pred=z['scores'].argmax(1)
    truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');y=np.array(truth[ids[loc['diagnostic']]],copy=True)
    value=float(np.mean([np.mean(pred[y==k]==k) for k in np.unique(y)]));g=c['pilot_diagnostic_gate']
    ref=read(c['source1_reference_summary'])['scores'][c['source1_reference_arm']];assert ref==g['reference']
    passed=value+g['tolerance']>=ref+g['minimum_gain']
    result=dict(state='complete_exploratory_Focal_GAP64_source1_pilot',diagnostic_macro=value,reference_noMixup_GAP64_macro=ref,difference=value-ref,predeclared_gate_passed=passed,route='separately_preregister_same_protocol_source6_confirmation' if passed else 'close_exact_Focal_alpha.25_gamma2_meanB12_GAP64_B64_seed20261029_no_rescue',users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,legacy8fold_improvement_proven=False,selected_epoch=frozen['selected_epoch'],finalizer_invocations=1)
    write(B/'SUMMARY.json',result);write(B/'STATE.json',dict(state=result['route'],diagnostic_labels_read=True,users7_8_labels_used=False,finalizer_consumed=True,summary_sha256=sha(B/'SUMMARY.json')));print(json.dumps(result))
if __name__=='__main__':main()
