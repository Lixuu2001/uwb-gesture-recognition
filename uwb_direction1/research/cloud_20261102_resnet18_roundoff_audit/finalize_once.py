from pathlib import Path
from completion_common import B,O,complete,read,write,sha,private_scope
from backup import core
from metrics import gate
def supported_macro(np,y,pred,expected):
 assert set(np.unique(y).tolist())==set(expected) and y.shape==pred.shape
 value=float(np.mean([np.mean(pred[y==k]==k) for k in np.unique(y)]));assert np.isfinite(value);return value
def main():
 c=complete();p=O/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json');core().verify_members(p/'remote_reassembled.tar.gz',m['members_manifest_sha256'],m['tracked_payload_files']);bound={x['path']:x['sha256'] for x in read(p/'payload/MEMBERS.json')['members']}
 from engine import np,assert_payload
 private_scope(m,v,r,bound)
 for name in ['COMPLETE_AUDIT.json','run01/native_fitted.pkl','run01/final_fitted.pkl','run01/ENCODER_STATE.json','run01/validation_unscored.npz','run01/diagnostic_unscored.npz']:assert bound[name]==sha(O/name if name=='COMPLETE_AUDIT.json' else B/name)
 assert not (B/'run01/SUMMARY.json').exists()
 from common import load_inputs
 import pickle
 ids,users,loc,maps,_=load_inputs(c,False);labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={}
 with (B/'run01/final_fitted.pkl').open('rb') as f:payload=pickle.load(f)
 assert_payload(payload,True)
 for role,u in [('validation',2),('diagnostic',1)]:
  with np.load(B/'run01'/(role+'_unscored.npz')) as z:
   np.testing.assert_array_equal(z['ids'],ids[loc[role]]);assert set(users[loc[role]].tolist())=={u};np.testing.assert_array_equal(z['predictions'],payload['head'].classes_[z['scores'].argmax(1)]);y=np.array(labels[z['ids']],copy=True)
   scores[role]=supported_macro(np,y,z['predictions'],range(10) if role=='validation' else range(12))
 result=dict(state='complete_exploratory_source_held1_full_ImageNet_ResNet18_frozen512_LinearSVC_C1',scores=scores,fixed_reference_macro=c['reference_held1_macro'],complete_private_shared_encoder_and_head_modelset_verified=True,source6_confirmation_complete=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,numerical_audit_contract_corrected_after_unlabelled_input_observation=True,original_failed_audit_preserved=True,model_or_feature_or_head_configuration_changed=False,**gate(scores['diagnostic'],c['reference_held1_macro'],c['minimum_gain'],c['tolerance']))
 write(B/'run01/SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
