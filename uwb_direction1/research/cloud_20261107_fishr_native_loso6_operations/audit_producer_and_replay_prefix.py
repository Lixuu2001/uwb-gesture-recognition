"""Read-only completed Fishr producer for any frozen LOSO fold and committed fresh replay prefix audit. No labels."""
from pathlib import Path
import sys,hashlib,json,time
P=Path('/workspace/uwb-fishr-native-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import B,read,write,sha
from controller import identity

def main(h):
 start=time.monotonic();cpu=time.process_time();assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 for n,expected_sha in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==expected_sha,n
 phase=read(P/'PHASE_IDENTITY.json');now=identity(phase['pid']);assert now and now['state']!='Z' and all(now[k]==phase[k] for k in ['pid','startticks','boot_id'])
 assert h in range(1,7) and B==(P/f'folds/held{h}').resolve()
 f=P/f'folds/held{h}';s=f/'run/train';r=f/'run/replay';cfg=read(f/'PROTOCOL.json');done=read(s/'COMPLETE.json');term=read(f/'run/execution/train/TERMINATION.json');expected=cfg['expected_updates']
 assert done['epochs']==64 and done['updates']==expected and term['exit_code']==0 and term['resource_stop_reason'] is None
 import engine as e
 from journal import load
 from loso_roles_metrics import roles
 state,progress=load(s);assert e.semantic(state)==done['final_state_sha256'] and state['epoch']==64 and state['updates']==expected
 opt=state['optimizer'];assert opt['algorithm_updates']==expected and opt['reset_events']==[1500] and opt['native_optimizer']['count']==expected-1500
 ema=opt['fishr_EMA'];assert ema['ema']==.95 and len(ema['data'])==len(ema['updates'])==4
 assert all(sum(v.numel() for v in group.values())==18060 for group in ema['data'])
 assert all(e.torch.isfinite(v).all() and not v.requires_grad for group in ema['data'] for v in group.values())
 selected=e.torch.load(s/'selected.pt',map_location='cpu',weights_only=True)
 assert selected['selected_epoch']==state['best_epoch']==done['selected_epoch'] and e.semantic(selected['model'])==e.semantic(state['best_model'])
 keys=[]
 for epoch in range(1,65):
  rec=read(s/f'epoch{epoch:03}.json');keys.append((rec['validation_macro'],-rec['validation_loss'],-epoch))
 assert state['best_epoch']==max(range(1,65),key=lambda ep:keys[ep-1]) and list(state['best_key'])==list(keys[state['best_epoch']-1])
 for name,key in [('selected.pt','selected_sha256'),('validation_unscored.npz','validation_outputs_sha256'),('diagnostic_unscored.npz','diagnostic_outputs_sha256')]:assert sha(s/name)==done[key]
 # Metadata only; never load UWB maps or gesture truth.
 assert sha(cfg['ids'])==cfg['ids_sha256']
 with e.np.load(cfg['ids']) as z:ids=z['ids'];users=z['users']
 _,loc=roles(users,h);assert len(ids)==6900 and [len(v) for v in loc.values()]==cfg['counts']
 with e.np.load(f/'roles.npz') as z:
  for name,ix in loc.items():e.np.testing.assert_array_equal(z[name],ids[ix])
 rows=[json.loads(line) for line in (s/'steps.jsonl').read_bytes().splitlines()];assert len(rows)==expected
 counts=[0]*4;resets=[];i=0
 for epoch in range(1,65):
  order=e.order(len(loc['train']),epoch)
  for offset in range(0,len(order),64):
   pos=order[offset:offset+64];row=rows[i];before=i;i+=1
   assert row['epoch']==epoch and row['offset']==offset and row['update']==i and row['completed_updates_before']==before and row['completed_updates_after']==i
   assert row['positions_sha256']==hashlib.sha256(pos.tobytes()).hexdigest() and row['training_ids_sha256']==hashlib.sha256(ids[loc['train'][pos]].tobytes()).hexdigest()
   present=set(users[loc['train'][pos]].tolist());assert present<=set(cfg['train_users']) and row['valid_training_domains']==len(present)
   for j,u in enumerate(cfg['train_users']):counts[j]+=u in present
   assert row['fishr_EMA_updates']==counts and row['classifier_gradient_coordinates']==18060 and row['forward_batch']==len(pos)
   assert row['classification_loss']==row['global_BCE'] and row['fishr_penalty_weight']==(0 if before<1500 else 1000)
   assert row['optimizer_reset_this_update']==(before==1500) and row['optimizer_reset_events']==([] if before<1500 else [1500])
   if row['optimizer_reset_this_update']:resets.append(i)
 assert counts==ema['updates'] and resets==[1501]
 # Commit boundary fixes scope even if replay continues while this audit runs.
 rp=read(r/'PROGRESS.json');n=rp['updates'];prefix=(r/'steps.jsonl').read_bytes()[:rp['prefix_bytes']]
 assert len(prefix.splitlines())==n and hashlib.sha256(prefix).hexdigest()==rp['prefix_sha256']
 assert all(json.loads(line)==rows[i] for i,line in enumerate(prefix.splitlines()))
 for epoch in range(rp['epoch']+1):assert read(r/f'epoch{epoch:03}.json')==read(s/f'epoch{epoch:03}.json')
 proof=dict(state='completed64_producer_and_committed_fresh_replay_prefix_readonly_audited',observed_unix=time.time(),held=h,producer_complete64=True,producer_updates=expected,selected_epoch_validation_only=done['selected_epoch'],producer_all_sample_ID_role_order_hashes_verified=True,producer_global_BCE_and_Fishr_schedule_verified=True,producer_reset_actual_update=resets[0],final_inner_native_optimizer_steps=expected-1500,EMA_history_coordinates_per_domain=18060,EMA_updates_by_training_user=dict(zip(map(str,cfg['train_users']),counts)),producer_native_state_semantic_sha256=done['final_state_sha256'],producer_all65_records_present=True,committed_replay_epoch=rp['epoch'],committed_replay_updates=n,committed_replay_all_steps_and_epochs_bitwise_equal=True,complete_replay_at_snapshot=rp['epoch']==64,whole_cohort_complete=False,no_models_trained_or_inference_run_by_audit=True,real_UWB_map_or_gesture_label_values_read=False,users7_8_labels_used=False,phase_actual_identity=now,resource_observation=read(P/'RESOURCE_LEDGER.json'),audit_wall=time.monotonic()-start,audit_CPU=time.process_time()-cpu,full_replay_alone_not_proof_no_shared_implementation_error=True)
 write(P/f'HELD{h}_PRODUCER_AND_REPLAY_PREFIX_AUDIT.json',proof);print(json.dumps(proof))
if __name__=='__main__':
 assert len(sys.argv)==2;main(int(sys.argv[1]))
