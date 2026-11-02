"""Read-only completed training/ID/artifact receipt; not the full replay audit.

Does not parse diagnostic SUMMARY, predictions, gesture labels or raw inputs,
and performs no real forward/optimizer step. The existing full auditor stays
authoritative for accuracy and all-state numerical replay.
"""
from pathlib import Path
import datetime,hashlib,json,os,time
B=Path('/workspace/uwb-hinception-pilot-20261031')
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();state=read(B/'run01/STATE.json');term=read(B/'supervisor01/TERMINATION.json');cfg=read(B/'PROTOCOL.json');done=read(B/'run01/TRAINING_COMPLETE.json');run=B/'run01'
 assert state['state']=='complete' and state['pid'] is None and state['epoch']==100 and state['total_steps']==7400 and state['all_predictions_frozen'] and state['users7_8_labels_used'] is False
 assert term['exit_code']==0 and not term['wall_timeout'] and term['supervisor_wall_seconds']<43200 and term['child_cpu_seconds']<43200
 frozen=read(B/'FROZEN_HASHES.json');artifacts=read(run/'FILES.json')
 for rel,d in frozen.items():assert sha(B/rel)==d,rel
 for rel,d in artifacts.items():assert sha(run/rel)==d,rel  # SUMMARY bytes hashed only
 for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]:assert artifacts[name]==done[key]
 predictions=read(run/'ALL_PREDICTIONS_FROZEN.json');assert set(predictions)=={'validation','diagnostic'}
 for role,d in predictions.items():assert artifacts[role+'_predictions_unscored.npz']==d
 split=read(run/'split.json');training_ids=split['train'];assert len(training_ids)==4700 and len(set(training_ids))==4700
 assert len(split['validation'])==1000 and len(split['diagnostic'])==1200
 assert not (set(training_ids)&set(split['validation']) or set(training_ids)&set(split['diagnostic']) or set(split['validation'])&set(split['diagnostic']))
 import torch
 torch.set_num_threads(1);rows=[json.loads(s) for s in (run/'iterations.jsonl').read_text().splitlines()];assert len(rows)==7400
 for ep in range(1,101):
  order=torch.randperm(4700,generator=torch.Generator().manual_seed(cfg['seed']+ep)).tolist();seen=[]
  for batch in range(74):
   row=rows[(ep-1)*74+batch];expected=[training_ids[i] for i in order[batch*64:(batch+1)*64]]
   assert row['epoch']==ep and row['batch']==batch and row['total_steps']==(ep-1)*74+batch+1 and row['training_ids']==expected
   assert len(expected)==(28 if batch==73 else 64);seen.extend(expected)
  assert len(seen)==len(set(seen))==4700 and set(seen)==set(training_ids)
 epochs=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines()];assert len(epochs)==100
 best_ep=None;best_loss=None
 for ep,row in enumerate(epochs,1):
  assert row['epoch']==ep and row['steps']==ep*74 and row['training_records']==4700
  loss=row['metrics']['loss']
  if best_loss is None or loss<best_loss:best_ep=ep;best_loss=loss
  assert row['best_epoch']==best_ep and row['best_training_loss']==best_loss
 assert (best_ep,best_loss)==(done['best_epoch'],done['best_training_loss'])==(state['best_epoch'],state['best_training_loss'])
 post=read(B/'POSTPROCESS_STATE.json');assert post['state']=='completed_run_audit_running' and Path('/proc',str(post['pid'])).exists()
 audit_supervisors=[]
 for child in Path('/proc',str(post['pid']),'task',str(post['pid']),'children').read_text().split():
  p=Path('/proc',child);cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'launch.py --audit' in cmd
  workers=p.joinpath('task',child,'children').read_text().split();assert len(workers)==1
  worker=Path('/proc',workers[0]);assert 'audit_complete.py --run' in worker.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();audit_supervisors.append(dict(supervisor_pid=int(child),worker_pid=int(workers[0])))
 assert len(audit_supervisors)==1 and not (B/'COMPLETE_AUDIT.json').exists()
 result=dict(state='completed_training_artifacts_and_all_ID_metadata_verified_full_numerical_audit_pending',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_termination=term,epochs=100,updates=7400,training_ID_appearances_verified=470000,each_epoch4700_unique_and_all_source_sampler_orders_exact=True,last_batch28_verified=True,source_strict_min_training_loss_selected_epoch=best_ep,all1423_frozen_sha256_verified=len(frozen)==1423,all_run_artifact_sha256_verified=artifacts,prediction_archives_frozen_sha256=predictions,all_model_state_numerical_replay_complete=False,accuracy_or_originality_claim=False,diagnostic_summary_parsed=False,raw_inputs_or_gesture_label_arrays_opened=False,prediction_arrays_opened=False,new_real_forward_fit_or_configuration=False,existing_postprocess_pid=post['pid'],existing_audit_processes=audit_supervisors,complete_private_backup_done=False,wall_seconds=time.monotonic()-start,verifier_sha256=sha(__file__))
 (OUT/'TRAINING_ARTIFACTS_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['state','epochs','updates','source_strict_min_training_loss_selected_epoch','existing_audit_processes','diagnostic_summary_parsed','wall_seconds']}))
if __name__=='__main__':main()
