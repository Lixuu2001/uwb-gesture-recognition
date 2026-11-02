from pathlib import Path
from common import B,complete,read,write,sha
from backup import core
from metrics import gate
def supported_macro(np,y,pred,expected):
    assert set(np.unique(y).tolist())==set(expected) and y.shape==pred.shape
    result=float(np.mean([np.mean(pred[y==k]==k) for k in np.unique(y)]))
    assert np.isfinite(result)
    return result
def main():
    c=complete();p=B/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json')
    assert m['scope']==v['scope']==r['scope']=='complete_audited_Arsenal25x2000_native_CVaccuracy_source_held1'
    assert v['complete_modelset_verified'] and r['complete_modelset_verified'] and r['numeric_audit_certified'] and r['state']=='private_readback_verified'
    assert m['archive_sha256']==v['archive_sha256']==r['archive_sha256'];core().verify_members(p/'remote_reassembled.tar.gz',m['members_manifest_sha256'],m['tracked_payload_files'])
    bound={x['path']:x['sha256'] for x in read(p/'payload/MEMBERS.json')['members']}
    for name in ['COMPLETE_AUDIT.json','run01/native_fitted.pkl','run01/final_fitted.pkl','run01/validation_unscored.npz','run01/diagnostic_unscored.npz']:assert bound[name]==sha(B/name)
    assert not (B/'run01/SUMMARY.json').exists()
    import sys
    sys.path.insert(0,c['source_preparation']);from native_arsenal import np
    from common import load_inputs
    ids,users,loc,maps,_=load_inputs(c,False);labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={}
    for role,u in [('validation',2),('diagnostic',1)]:
        with np.load(B/'run01'/(role+'_unscored.npz')) as z:
            np.testing.assert_array_equal(z['ids'],ids[loc[role]]);assert set(users[loc[role]].tolist())=={u}
            from native_arsenal import canonical_predictions
            import pickle
            with (B/'run01/final_fitted.pkl').open('rb') as f:model=pickle.load(f)
            np.testing.assert_array_equal(z['predictions'],canonical_predictions(model,z['probabilities']));y=np.array(labels[z['ids']],copy=True)
            scores[role]=supported_macro(np,y,z['predictions'],range(10) if role=='validation' else range(12))
    result=dict(state='complete_exploratory_source_held1_Arsenal25x2000_native_CVaccuracy',scores=scores,fixed_reference_macro=c['reference_held1_macro'],complete_private_modelset_verified=True,source6_confirmation_complete=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,**gate(scores['diagnostic'],c['reference_held1_macro'],c['minimum_gain'],c['tolerance']))
    write(B/'run01/SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
