"""Protect an inactive completed64 producer while original own-state replay runs; never score."""
from pathlib import Path
import sys,json,time
P=Path('/workspace/uwb-vrex-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from controller import identity
from stream_archive import build,verify

def create(h):
 started=time.monotonic();cpu=time.process_time();assert h in range(1,7)
 phase=read(P/'PHASE_IDENTITY.json');now=identity(phase['pid']);assert now and now['state']!='Z' and all(phase[k]==now[k] for k in ['pid','boot_id','startticks'])
 ledger=read(P/'RESOURCE_LEDGER.json');assert ledger['active_held']==h and ledger['fold_stage']=='replay' and ledger['wall']<43200 and ledger['cpu']<43200
 f=P/f'folds/held{h}';s=f/'run/train';out=P/f'producer_backups/held{h}';assert not out.exists()
 old=read(f/'run/execution/train/IDENTITY.json');actual=identity(old['pid']);assert not actual or actual['state']=='Z' or any(old[k]!=actual[k] for k in ['boot_id','startticks'])
 term=read(f/'run/execution/train/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
 assert read(f/'run/execution/train/FIRST_AUDIT_TERMINATION.json')['exit_code']==0
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 for n,v in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==v,n
 for n,v in read(P/'shared/DEPENDENCIES.json')['files'].items():assert sha(n)==v,n
 c=read(f/'PROTOCOL.json');done=read(s/'COMPLETE.json');assert done['epochs']==64 and done['updates']==c['expected_updates'] and done['diagnostic_labels_read'] is False
 import engine as e
 from journal import load as load_journal
 state,progress=load_journal(s);assert state['epoch']==64 and state['updates']==done['updates'] and e.semantic(state)==done['final_state_sha256']
 assert sha(s/'final_full.pt')==progress['checkpoint_sha256']
 selected=e.torch.load(s/'selected.pt',map_location='cpu',weights_only=True);assert selected['selected_epoch']==state['best_epoch'] and e.semantic(selected['model'])==e.semantic(state['best_model'])
 for name,key in [('selected.pt','selected_sha256'),('validation_unscored.npz','validation_outputs_sha256'),('diagnostic_unscored.npz','diagnostic_outputs_sha256')]:assert sha(s/name)==done[key]
 rows=(s/'steps.jsonl').read_bytes();assert len(rows.splitlines())==done['updates'] and all(json.loads(x)['update']==i+1 for i,x in enumerate(rows.splitlines()))
 proof=dict(state='completed64_producer_native_state_selected_model_journal_and_committed_prefix_verified_before_private_protection',held=h,epochs=64,updates=done['updates'],selected_epoch=done['selected_epoch'],final_native_state_sha256=done['final_state_sha256'],journal_chain_and_step_prefix_verified=True,native_first_update_exact=True,complete_training_producer=True,complete_independent_replay=False,complete_fold_audits=False,complete_sixfold_cohort=False,diagnostic_labels_read=False,users7_8_labels_used=False,phase=now,replay_ledger=ledger,producer_termination=term,original_phase_not_restarted=True,new_model_updates_or_inference_executed=False)
 write(P/f'PRODUCER_{h}_BACKUP_AUDIT.json',proof)
 files={}
 for q in s.rglob('*'):
  if q.is_file() and '__pycache__' not in q.parts and not q.name.endswith(('.lock','.tmp')):files['producer/'+str(q.relative_to(s))]=q
 for n in ['PROTOCOL.json','roles.npz']:files['fold/'+n]=f/n
 for n in ['IDENTITY.json','FIRST_AUDIT_IDENTITY.json','FIRST_AUDIT_TERMINATION.json','TERMINATION.json','stdout.log','first_audit.log']:files['execution/train/'+n]=f/'run/execution/train'/n
 for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REAL_RELEASE.json','REMOTE_PREREGISTRATION.json','PRIOR_CLOSURE_AND_REUSE.json']:files['phase/'+n]=P/n
 for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
 for n in ['CLOUD_DOWNLOAD_INDEX.json','MANIFEST.json','READBACK_VERIFIED.json']:files['private_initial/'+n]=P/'initial_backup'/n
 files['PRODUCER_BACKUP_AUDIT.json']=P/f'PRODUCER_{h}_BACKUP_AUDIT.json';files['operator/producer_archive.py']=Path(__file__)
 layout=P/f'PRODUCER_{h}_SOURCE_LAYOUT.json';write(layout,dict(archive_to_expected_path={n:str(q) for n,q in files.items()},raw_arrays_and_runtime_external=True,standalone_auto_resume_authorized=False));files['SOURCE_LAYOUT.json']=layout
 out.parent.mkdir(parents=True,exist_ok=True);m=build(out,f'vrex_loso6_producer_held{h}_20261106',files,'completed64_producer_only_fresh_replay_and_full_fold_audit_pending_unscored',chunk_size=7340032,limit=134217728)
 m.update(state='complete_producer_only_streamed_independent_replay_pending_unscored',complete_modelset=False,complete_training_producer=True,complete_independent_replay=False,complete_fold_audits=False,complete_sixfold_cohort=False,held=h,archive_total_build_wall=time.monotonic()-started,archive_total_build_CPU=time.process_time()-cpu);write(out/'MANIFEST.json',m);return m

def check(h):
 started=time.monotonic();cpu=time.process_time();d=P/f'producer_backups/held{h}';m=read(d/'MANIFEST.json');assert m['held']==h and m['complete_training_producer'] and not m['complete_modelset']
 rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
 for r in rows:
  q=r['metadata'];assert q['shared'] is False and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
  assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
 v=verify(m,rows);assert v['complete_modelset_verified'] is False;write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
 write(d/'READBACK_COST.json',dict(wall=time.monotonic()-started,CPU=time.process_time()-cpu,scope='whole_archive_decompression_and_allmember_hash_verification'))
 write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(state='completed64_training_producer_only_owner_only_raw_allmember_verified_independent_replay_pending',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,held=h,complete_training_producer=True,complete_independent_replay=False,complete_fold_audits=False,complete_sixfold_cohort=False,automatic_resume_authorized=False,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows]));return v
if __name__=='__main__':print(json.dumps(create(int(sys.argv[2])) if sys.argv[1]=='build' else check(int(sys.argv[2]))))
