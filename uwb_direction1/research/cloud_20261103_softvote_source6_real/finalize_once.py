"""Only all-five-fold completion and private recovery authorize one aggregate."""
from pathlib import Path
import os
from common import B,read,write,sha,privacy
from archive import all_frozen
from stream_archive import verify
from roles_gate import improvement_gate
def require_final_private():
    c,p=all_frozen();d=B/'full_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json');r=read(d/'DRIVE_BACKUP_RECEIPT.json');rows=read(d/'READBACK_INPUT.json')
    privacy(rows);assert verify(m,rows)==v
    assert r['state']=='private_raw_all_members_verified' and r['complete_modelset_verified']
    assert r['archive_sha256']==m['archive_sha256']==v['archive_sha256']
    assert m['scope']==r['scope']==c['full_archive_scope'] and not r['diagnostic_labels_scored']
    assert sha(d/'MEMBERS.json')==m['members_manifest_sha256']==r['members_manifest_sha256']
    members={x['path']:x for x in read(d/'MEMBERS.json')['members']}
    for held in [2,3,4,5,6]:
        for arm in ['timemil','shuffle']:
            name=f'folds/held{held}/{arm}/selected.pt';assert members[name]['sha256']==sha(B/name)
        name=f'folds/held{held}/fusion/diagnostic_unscored.npz';assert members[name]['sha256']==sha(B/name)
    return c,p
def main():
    c,p=require_final_private();assert not (B/'SUMMARY.json').exists()
    fd=os.open(B/'FINALIZE_ONCE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    import numpy as np
    from data import load,macro
    pilot=read(Path(p['reused_held1_root'])/'SUMMARY.json')
    source1_tm=read(c['held1_member_summaries']['timemil']['path'])['scores']['timemil']
    source1_sh=read(c['held1_member_summaries']['shuffle']['path'])['scores']['diagnostic']
    rows=[dict(held=1,timemil=source1_tm,shuffle=source1_sh,mixture=pilot['diagnostic_macro'],reused_without_rescoring=True)]
    for held in [2,3,4,5,6]:
        ids,users,loc,maps,_,_=load(held,labels=False)
        with np.load(B/'folds'/f'held{held}'/'fusion/diagnostic_unscored.npz') as z:
            np.testing.assert_array_equal(z['ids'],ids[loc['diagnostic']]);pred={k:z[k] for k in ['timemil_predictions','shuffle_predictions','predictions']}
        truth=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');y=np.array(truth[ids[loc['diagnostic']]],copy=True)
        rows.append(dict(held=held,timemil=macro(y,pred['timemil_predictions']),shuffle=macro(y,pred['shuffle_predictions']),mixture=macro(y,pred['predictions']),reused_without_rescoring=False))
    scores={arm:np.array([r[arm] for r in rows]) for arm in ['timemil','shuffle','mixture']};g=p['gate'];comparisons={}
    for arm in ['timemil','shuffle']:
        d=scores['mixture']-scores[arm]
        comparisons[arm]=dict(mean_difference=float(d.mean()),wins=int((d>g['tolerance']).sum()),worst=float(d.min()),gate_passed=improvement_gate(scores['mixture'],scores[arm],g['member_mean_gain'],g['member_minimum_strict_wins'],g['member_worst_lower'],g['tolerance']))
    for name,key in [('localCNN24','local'),('GAP64','gap')]:
        ref=read(c['baselines'][name]['SUMMARY.json']['path']);r=sorted(ref['rows'],key=lambda x:x['held']);assert [x['held'] for x in r]==list(range(1,7));values=np.array([x[key] for x in r]);d=scores['mixture']-values
        comparisons[name]=dict(mean_difference=float(d.mean()),wins=int((d>g['tolerance']).sum()),worst=float(d.min()),gate_passed=improvement_gate(scores['mixture'],values,g['strong_baseline_mean_gain'],g['strong_baseline_minimum_strict_wins'],g['strong_baseline_worst_lower'],g['tolerance']))
    passed=all(x['gate_passed'] for x in comparisons.values())
    result=dict(state='complete_exploratory_source6_mixture_and_component_controls',rows=rows,means={k:float(v.mean()) for k,v in scores.items()},comparisons=comparisons,all_locked_gates_passed=passed,route='separately_register_same_protocol_eightfold_benchmark' if passed else 'close_exact_source6_uniform_TimeMIL64_Shuffle100_T1_seed20261029_no_rescue',source1_scores_reused_without_rescoring=True,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,legacy8fold_improvement_proven=False)
    write(B/'SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
