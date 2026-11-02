"""Fresh full 40 epoch replay; same native kernels, separate process."""
from common import B,ready,read,write,sha,load_inputs
import json,time
def main():
 c=ready();start=time.monotonic();out=B/'run01';term=read(B/'supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['resource_stop_reason'] and not term['wall_timeout'] and term['cgroup_oom_kill_delta']==0
 for n,h in read(out/'FILES.json').items():assert sha(out/n)==h,n
 from engine import np,torch,make,run_training,semantic,evaluate
 ids,users,loc,maps,y,v=load_inputs(c)
 with np.load(out/'split.npz') as z:
  for k,a in dict(ids=ids,users=users,**loc).items():np.testing.assert_array_equal(z[k],a)
 m=make();saved=torch.load(out/'initial12_head.pt',map_location='cpu',weights_only=True);assert semantic(m.fc.state_dict())==semantic(saved);del m,saved
 with (out/'steps.jsonl').open() as steps:
  def on_step(r):assert json.loads(next(steps))==r,'trainingstep/order/loss/RNGmismatch'
  def on_epoch(r,scores,state,best,improved):
   assert read(out/f"epoch{r['epoch']:02d}.json")==r
   with np.load(out/f"validation_epoch{r['epoch']:02d}.npz") as z:np.testing.assert_array_equal(scores,z['scores']);np.testing.assert_array_equal(z['predictions'],scores.argmax(1));np.testing.assert_array_equal(z['ids'],ids[loc['validation']])
   print(json.dumps(dict(stage='full_replay',epoch=r['epoch'])),flush=True)
  m,opt,best,epoch,acc,initial=run_training(maps,loc,y,v,on_step,on_epoch)
  assert next(steps,None) is None
 final=torch.load(out/'final_full.pt',map_location='cpu',weights_only=True);assert final['epoch']==40 and final['updates']==12560;assert semantic(final['model'])==semantic(m.state_dict()) and semantic(final['optimizer'])==semantic(opt.state_dict());selection=read(out/'TRAIN_SELECTION.json');assert selection['selected_epoch']==epoch and selection['best_validation_macro']==acc and selection['initial_full_model_state_sha256']==initial
 last=read(out/'epoch40.json')
 for key in ['order_rng','augmentation_rng','torch_rng']:assert semantic(final[key])==last[key+'_sha256']
 saved=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True);assert saved['epoch']==epoch and saved['validation_macro']==acc and semantic(saved['model'])==semantic(best);m.load_state_dict(saved['model'],strict=True)
 for role in ['validation','diagnostic']:
  scores=evaluate(m,maps,loc[role])
  with np.load(out/(role+'_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[loc[role]]);np.testing.assert_array_equal(z['scores'],scores);np.testing.assert_array_equal(z['predictions'],scores.argmax(1))
 from audit_preprocessing import verify
 preprocessing=verify(c,maps,ids)
 for n,h in read(out/'FILES.json').items():assert sha(out/n)==h,n
 write(B/'COMPLETE_AUDIT.json',dict(state='pass',full40_epoch12560_update_replay_exact=True,all40_validation_scores_selection_exact=True,best_full_model_and_final_model_optimizer_momentum_rng_exact=True,all2200_selected_scores_predictions_exact=True,diagnostic_labels_read=False,validation_labels_read_for_registered_checkpoint_selection=True,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,auditor_sha256=sha(__file__),artifact_hashes={**{str(out/n):h for n,h in read(out/'FILES.json').items()},str(out/'FILES.json'):sha(out/'FILES.json')},audit_wall_seconds=time.monotonic()-start,**preprocessing))
if __name__=='__main__':main()
