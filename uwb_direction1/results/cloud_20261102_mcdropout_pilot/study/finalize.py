from pathlib import Path
from guards import read,sha
from backup import complete,core
from artifacts import write
B=Path(__file__).resolve().parent
def main():
 c=complete();p=B/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json');assert m['scope']==v['scope']=='complete_audited_MC100_frozen_TimeMIL_inference' and v['complete_modelset_verified'] and r['complete_modelset_verified'] and r['state']=='private_readback_verified' and m['archive_sha256']==v['archive_sha256']==r['archive_sha256'];core().verify_members(p/'remote_reassembled.tar.gz',m['members_manifest_sha256'],m['tracked_payload_files'])
 bound={x['path']:x['sha256'] for x in read(p/'payload/MEMBERS.json')['members']};assert bound['COMPLETE_AUDIT.json']==sha(B/'COMPLETE_AUDIT.json')
 for n in ['validation_MC_unscored.npz','diagnostic_MC_unscored.npz','validation_deterministic_unscored.npz','diagnostic_deterministic_unscored.npz','initial_state.pt','first_mc_pass.pt','final_state.pt','native_selected_weights.pt']:assert bound['run01/'+n]==sha(B/'run01'/n)
 assert not (B/'run01/SUMMARY.json').exists()
 from engine import np,torch
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={};control={}
 def metric(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
 for role,u in [('validation',2),('diagnostic',1)]:
  with np.load(B/'run01'/(role+'_MC_unscored.npz')) as a:
   pi=a['ids'];dec=a['predictions'];mean=torch.sigmoid(torch.from_numpy(a['logits'])).numpy().mean(0,dtype='float64');np.testing.assert_array_equal(mean,a['mean_sigmoid']);np.testing.assert_array_equal(dec,mean.argmax(1))
  assert len(pi)==len(set(pi.tolist())) and set(pi.tolist())==set(ids[users==u].tolist());y=np.array(labels[pi],copy=True);scores[role]=metric(y,dec)
  with np.load(B/'run01'/(role+'_deterministic_unscored.npz')) as a:np.testing.assert_array_equal(a['ids'],pi);control[role]=metric(y,a['logits'].argmax(1))
 assert abs(control['diagnostic']-c['reference_held1_macro'])<=1e-12;delta=scores['diagnostic']-control['diagnostic'];passed=delta+c['gate']['tolerance']>=c['gate']['minimum_gain_over_fixed_reference']
 result=dict(state='complete_exploratory_source_held1_MC100_pilot',scores=scores,deterministic_controls=control,fixed_reference_macro=c['reference_held1_macro'],delta_macro=delta,predeclared_gate_passed=passed,route=c['gate']['pass_route'] if passed else c['gate']['fail_route'],samples_per_role=100,reused_selected_epoch=41,new_training=False,gradient_updates=0,complete_private_modelset_verified=True,source6_confirmation_complete=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False);write(B/'run01/SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
