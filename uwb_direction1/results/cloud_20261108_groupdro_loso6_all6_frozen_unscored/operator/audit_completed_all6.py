"""Read-only full cohort native/q/roles/replay audit before any held truth pairing."""
from pathlib import Path
import sys,json,time,copy,hashlib,datetime
P=Path('/workspace/uwb-groupdro-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
from journal import load as journal_load
from loso_roles_metrics import roles
import engine as e
from groupdro_core import NativeGroupDROOptimizer

def main():
 start=time.monotonic();cpu=time.process_time();ready(True)
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','boot_id','startticks'])
 phase=read(P/'STATE.json');assert phase['completed_folds']==[1,2,3,4,5,6] and phase['wall']<=43200 and phase['cpu']<=43200
 frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json')
 for r in frozen['files'].values():assert sha(r['path'])==r['sha256']
 with e.np.load(e.c['ids']) as z:ids=z['ids'];users=z['users']
 rows=[];all_artifacts={}
 for h in range(1,7):
  fold=P/f'folds/held{h}';cfg=read(fold/'PROTOCOL.json');assert cfg['train_users']==[u for u in range(1,7) if u not in [h,h%6+1]]
  val,loc=roles(users,h);assert val==cfg['validation_user'] and list(map(len,loc.values()))==cfg['counts']
  with e.np.load(fold/'roles.npz') as z:
   for role,ix in loc.items():e.np.testing.assert_array_equal(z[role],ids[ix])
  assert read(fold/'PHASE_TERMINATION.json')['exit_code']==0
  for stage in ['train','replay','output']:
   term=read(fold/f'run/execution/{stage}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
  assert read(fold/'run/execution/train/FIRST_AUDIT_TERMINATION.json')['exit_code']==0 and read(fold/'run/train/FIRST_RELEASE.json')['native_first_update_exact']
  first=read(fold/'run/train/FIRST_INDEPENDENT_REFERENCE.json');assert first['max_gradient_absolute_difference']<=2e-6 and first['max_q_absolute_difference']<=1e-7 and not first['diagnostic_labels_read']
  train=fold/'run/train';replay=fold/'run/replay';a=read(train/'COMPLETE.json');b=read(replay/'COMPLETE.json');assert a['epochs']==b['epochs']==64 and a['updates']==b['updates']==cfg['expected_updates']
  st,progress=journal_load(train);fresh,_=journal_load(replay)
  assert e.semantic(st)==e.semantic(fresh)==a['final_state_sha256']==b['final_state_sha256']
  assert e.semantic(e.torch.load(train/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
  assert e.semantic(e.torch.load(replay/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
  assert (train/'steps.jsonl').read_bytes()==(replay/'steps.jsonl').read_bytes()
  for epoch in range(65):assert read(train/f'epoch{epoch:03}.json')==read(replay/f'epoch{epoch:03}.json')
  m=e.native.make_model('gap',cfg['seed']);assert sum(p.numel() for p in m.parameters())==617612
  o=NativeGroupDROOptimizer(m.parameters(),cfg['train_users'],lambda ps:e.native.StableLookahead(ps,lr=.001,weight_decay=.0001));g=e.torch.Generator().manual_seed(cfg['seed'])
  e.restore(copy.deepcopy(st),m,o,g);assert o.native.count==o.robust.updates==cfg['expected_updates']
  assert e.semantic(e.capture(m,o,g,64,st['updates'],st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
  selected=e.torch.load(train/'selected.pt',map_location='cpu',weights_only=True)
  best=max((read(train/f'epoch{k:03}.json') for k in range(1,65)),key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
  assert selected['selected_epoch']==st['best_epoch']==a['selected_epoch']==b['selected_epoch']==best['epoch'] and e.semantic(selected['model'])==e.semantic(st['best_model'])
  for role in ['validation','diagnostic']:
   with e.np.load(train/(role+'_unscored.npz')) as zz,e.np.load(replay/(role+'_unscored.npz')) as ff:
    e.np.testing.assert_array_equal(zz['ids'],ids[loc[role]]);e.np.testing.assert_array_equal(ff['ids'],zz['ids']);e.np.testing.assert_array_equal(ff['scores'],zz['scores'])
  prev=e.np.ones(4,dtype=e.np.float64);maxq=0.;maxloss=0.;count=0
  for line in (train/'steps.jsonl').open():
   r=json.loads(line);count+=1;assert r['update']==r['completed_updates_after']==count and r['completed_updates_before']==count-1 and r['group_users']==cfg['train_users']
   cnt=e.np.array(r['domain_counts']);risk=e.np.array(r['source_risks']);q=e.np.array(r['group_q']);present=cnt>0
   assert cnt.sum()==r['forward_batch'] and list(present)==r['present'] and e.np.isfinite(risk).all() and e.np.isfinite(q).all() and (q>=0).all()
   expected=prev.copy();expected[present]*=e.np.exp(.01*risk[present]);expected/=expected.sum();maxq=max(maxq,float(abs(expected-q).max()));assert maxq<=1.5e-7
   w=q.copy()
   if not present.all():w[present]/=w[present].sum()
   maxloss=max(maxloss,abs(float(e.np.dot(w[present],risk[present]))-r['loss']));assert maxloss<=2e-6
   prev=q
  assert count==cfg['expected_updates'];e.np.testing.assert_array_equal(prev,st['optimizer']['groupDRO']['q'].numpy())
  for f in fold.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.tmp','.lock')):all_artifacts[str(f.relative_to(P))]=sha(f)
  rows.append(dict(held=h,epochs=64,production_updates=count,independent_fresh_replay_updates=count,all65_records_and_step_logs_bitwise_equal=True,native_q_optimizer_allRNG_mode_disk_restore_exact=True,role_ID_subject_disjoint=True,selected_epoch_validation_only=best['epoch'],first_reference=first,q_logged_transition_maxdiff=maxq,weighted_BCE_arithmetic_maxdiff=maxloss,held_labels_read=False))
  del st,fresh,m,o,g,selected
 # Reuse and verify the original strong comparator artifacts without any new truth pairing.
 base=read(P/'operator/INFERENCE_PROTOCOL.json')['baseline_artifacts']
 for r in base.values():assert sha(r['model'])==r['model_sha256'] and sha(r['prediction'])==r['prediction_sha256']
 result=dict(state='all6_native_models_q_optimizer_RNG_roles_selection_fullfreshreplay_outputs_frozen_unscored',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,production_updates=sum(r['production_updates'] for r in rows),fresh_replay_updates=sum(r['independent_fresh_replay_updates'] for r in rows),first_update_audit_extra_native_reproductions=6,frozen_code_sha256=sha(P/'FROZEN_HASHES.json'),frozen_code_file_count=len(read(P/'FROZEN_HASHES.json')),all_fold_artifact_sha256=all_artifacts,baseline_models_predictions_original_SHA_verified=True,source_review_scope_not_changed=True,diagnostic_labels_read=False,diagnostic_scores_compared_without_truth_for_exact_reproduction_only=True,no_new_forward_backward_optimizer_updates=True,logged_risk_q_check_is_not_independent_validation_of_risks_against_data=True,does_not_exclude_all_shared_implementation_errors=True,users7_8_labels_used=False,independent_validation=False,method_spending_gate_unchanged=read(P/'DEVELOPMENT_PROTOCOL.json')['bounded_LOSO_continuation_gate'],phase_cost=phase,audit_wall=time.monotonic()-start,audit_CPU=time.process_time()-cpu)
 write(P/'PRE_SCORE_FREEZE.json',result);write(P/'operator/ALL6_PRE_SCORE_AUDIT.json',result)
 print(json.dumps(dict(state=result['state'],folds=6,production_updates=result['production_updates'],fresh_replay_updates=result['fresh_replay_updates'],selected_epochs=[r['selected_epoch_validation_only'] for r in rows],artifacts=len(all_artifacts),audit_wall=result['audit_wall'],audit_CPU=result['audit_CPU'])))
if __name__=='__main__':main()
