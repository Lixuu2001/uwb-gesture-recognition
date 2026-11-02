"""One diagnostic score only after successful audit and complete private recovery."""
from pathlib import Path
from guards import read,sha
from backup import complete,core
from artifacts import write
B=Path(__file__).resolve().parent
def main():
 c=complete();p=B/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');receipt=read(p/'DRIVE_BACKUP_RECEIPT.json')
 assert m['scope']==v['scope']=='complete_audited_full20_modelset' and v['complete_modelset_verified'] and receipt['complete_modelset_verified'] and receipt['state']=='private_readback_verified' and m['archive_sha256']==v['archive_sha256']==receipt['archive_sha256']
 core().verify_members(p/'remote_reassembled.tar.gz',m['members_manifest_sha256'],m['tracked_payload_files'])
 bound={r['path']:r['sha256'] for r in read(p/'payload/MEMBERS.json')['members']};assert bound['COMPLETE_AUDIT.json']==sha(B/'COMPLETE_AUDIT.json')
 for name in ['diagnostic_predictions_unscored.npz','validation_predictions_unscored.npz','final_prediction_state.pt','epoch10.pt','last.pt']:assert bound['run01/'+name]==sha(B/'run01'/name)
 assert not (B/'run01/SUMMARY.json').exists()
 from engine import np
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 labels=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');scores={}
 def metric(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
 for role,u in [('validation',2),('diagnostic',1)]:
  with np.load(B/'run01'/(role+'_predictions_unscored.npz')) as z:pi=z['ids'];decision=z['predictions'];assert np.array_equal(decision,z['log_probabilities'].argmax(1))
  assert len(pi)==len(set(pi.tolist())) and set(pi.tolist())==set(ids[users==u].tolist());y=np.array(labels[pi],copy=True);scores[role]=metric(y,decision)
  if role=='diagnostic':diagnostic_ids=pi;diagnostic_y=y
 assert sha(c['reference_predictions'])==c['reference_predictions_sha256']
 with np.load(c['reference_predictions']) as z:
  ri=z['ids'];rp=z['logits'].argmax(1)
 assert set(ri.tolist())==set(diagnostic_ids.tolist());ref=metric(np.array(labels[ri],copy=True),rp);assert abs(ref-c['reference_held1_macro'])<=1e-12
 delta=scores['diagnostic']-ref;passed=delta+c['gate']['tolerance']>=c['gate']['minimum_gain_over_fixed_reference']
 result={'state':'complete_exploratory_source_held1_pilot','scores':scores,'fixed_reference_macro':ref,'delta_macro':delta,'predeclared_gate_passed':passed,'route':c['gate']['pass_route'] if passed else c['gate']['fail_route'],'H_historical_held1_context_macro':c['historical_H_held1_context_macro'],'selected_epoch':read(B/'run01/TRAINING_COMPLETE.json')['selected_epoch'],'complete_private_modelset_verified':True,'source6_confirmation_complete':False,'users7_8_labels_used':False,'independent_validation':False,'novel_contribution_validated':False}
 write(B/'run01/SUMMARY.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
