"""Read-only backup of one already completed native Fishr LOSO fold, never score/refit."""
from pathlib import Path
import sys,json,time,resource
P=Path('/workspace/uwb-fishr-native-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from controller import identity
from stream_archive import build,verify

def create(h):
 start=time.monotonic();cpu=time.process_time();assert h in range(1,7)
 f=P/f'folds/held{h}';c=read(f/'PROTOCOL.json');phase=read(P/'REAL_RELEASE.json')
 assert phase['branch_head_verified'] and phase['frozen_sha256']==sha(P/'FROZEN_HASHES.json')
 for n,v in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==v,n
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 assert read(f/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
 old=read(f/'CONTROLLER_IDENTITY.json');now=identity(old['pid']);assert not now or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
 assert read(f/'PHASE_TERMINATION.json')['exit_code']==0
 for stage in ['train','replay','output']:
  q=read(f/f'run/execution/{stage}/TERMINATION.json');assert q['exit_code']==0 and q['resource_stop_reason'] is None
 train=read(f/'run/train/COMPLETE.json');rep=read(f/'run/replay/COMPLETE.json')
 assert train['epochs']==rep['epochs']==64 and train['updates']==rep['updates']==c['expected_updates']
 assert train['final_state_sha256']==rep['final_state_sha256'] and not train['diagnostic_labels_read'] and not rep['diagnostic_labels_read']
 for ep in range(65):assert read(f/f'run/train/epoch{ep:03}.json')==read(f/f'run/replay/epoch{ep:03}.json')
 assert (f/'run/train/steps.jsonl').read_bytes()==(f/'run/replay/steps.jsonl').read_bytes()
 frozen=read(f/'run/PREDICTIONS_FROZEN.json')
 for n,v in frozen['files'].items():assert sha(f/'run'/n)==v,n
 d=P/f'completed_fold_backups/held{h}';assert not d.exists()
 files={}
 for q in f.rglob('*'):
  if q.is_file() and '__pycache__' not in q.parts and not q.name.endswith(('.lock','.tmp')):files['fold/'+str(q.relative_to(f))]=q
 for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REAL_RELEASE.json','REMOTE_PREREGISTRATION.json','PRIOR_CLOSURE_AND_REUSE.json']:files['phase/'+n]=P/n
 for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
 for n in ['CLOUD_DOWNLOAD_INDEX.json','MANIFEST.json','READBACK_VERIFIED.json']:files['private_initial/'+n]=P/'initial_backup'/n
 proof=dict(state='one_completed_fold_all64_epochs_all_updates_and_native_outputs_reaudited_blind_before_private_backup',held=h,epochs=64,updates=c['expected_updates'],all65_epoch_records_equal=True,all_steps_bitwise_equal=True,selected_epoch=train['selected_epoch'],native_final_state_sha256=train['final_state_sha256'],whole_cohort_complete=False,diagnostic_labels_read=False,users7_8_labels_used=False,archive_preparation_wall=time.monotonic()-start,archive_preparation_CPU=time.process_time()-cpu)
 q=P/f'COMPLETED_FOLD_{h}_BACKUP_AUDIT.json';write(q,proof);files['FOLD_BACKUP_AUDIT.json']=q
 write(P/f'COMPLETED_FOLD_{h}_SOURCE_LAYOUT.json',dict(archive_to_expected_path={n:str(q) for n,q in files.items()},initial_and_raw_array_recovery_external=True));files['SOURCE_LAYOUT.json']=P/f'COMPLETED_FOLD_{h}_SOURCE_LAYOUT.json'
 d.parent.mkdir(parents=True,exist_ok=True)
 m=build(d,f'fishr_native_loso6_completed_held{h}_20261107',files,f'complete_native_unscored_Fishr_held{h}_only_not_complete_sixfold_cohort',chunk_size=7340032,limit=268435456)
 m.update(state='single_completed_fold_native_modelset_streamed_unscored',complete_modelset=True,complete_single_fold=True,complete_sixfold_cohort=False,held=h,archive_total_build_wall=time.monotonic()-start,archive_total_build_CPU=time.process_time()-cpu);write(d/'MANIFEST.json',m);return m

def check(h):
 d=P/f'completed_fold_backups/held{h}';m=read(d/'MANIFEST.json');assert m['complete_single_fold'] and not m['complete_sixfold_cohort'] and m['held']==h
 rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
 for r in rows:
  q=r['metadata'];assert q['shared'] is False and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
  assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
 v=verify(m,rows);assert v['complete_modelset_verified'];write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
 write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(state='one_completed_unscored_Fishr_fold_owner_only_raw_whole_archive_allmember_verified',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,held=h,complete_single_fold=True,complete_sixfold_cohort=False,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows]))
 return v
if __name__=='__main__':print(json.dumps(create(int(sys.argv[2])) if sys.argv[1]=='build' else check(int(sys.argv[2]))))
