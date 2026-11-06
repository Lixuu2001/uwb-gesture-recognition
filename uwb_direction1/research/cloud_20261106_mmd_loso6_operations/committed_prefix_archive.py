"""Read-only generic checkpoint-prefix protection; no fit, inference, scoring or restart."""
from pathlib import Path
import sys,json,shutil,time,hashlib
START=time.monotonic();CPU_START=time.process_time()
P=Path('/workspace/uwb-mmd-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from stream_archive import build,verify
from controller import identity
import engine as e

def create(h,stage):
 assert h in range(2,7) and stage in ['train','replay']
 release=read(P/'REAL_RELEASE.json');assert release['frozen_sha256']==sha(P/'FROZEN_HASHES.json') and release['branch_head_verified']
 for n,v in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==v,n
 for n,v in read(P/'shared/DEPENDENCIES.json')['files'].items():assert sha(n)==v,n
 phase=read(P/'PHASE_IDENTITY.json');now=identity(phase['pid']);assert now and now['state']!='Z' and all(phase[k]==now[k] for k in ['pid','boot_id','startticks'])
 ledger=read(P/'RESOURCE_LEDGER.json');assert ledger['active_held']==h and ledger['fold_stage']==stage
 assert ledger['wall']<43200 and ledger['cpu']<43200 and not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 f=P/f'folds/held{h}';s=f/'run'/stage;assert not(s/'COMPLETE.json').exists()
 d=P/f'progress_snapshots/held{h}_{stage}';out=P/f'progress_backups/held{h}_{stage}';assert not d.exists() and not out.exists()
 raw=(s/'PROGRESS.json').read_bytes();progress=json.loads(raw);assert progress['epoch']>=1
 d.mkdir(parents=True);shutil.copyfile(s/progress['slot'],d/'committed_full_state.pt');assert sha(d/'committed_full_state.pt')==progress['checkpoint_sha256']
 prefix=(s/'steps.jsonl').read_bytes()[:progress['prefix_bytes']];assert hashlib.sha256(prefix).hexdigest()==progress['prefix_sha256'];lines=prefix.splitlines();assert len(lines)==progress['updates']
 assert all(json.loads(line)['update']==i+1 for i,line in enumerate(lines));(d/'steps_prefix.jsonl').write_bytes(prefix);(d/'PROGRESS.json').write_bytes(raw)
 previous=None
 for epoch in range(progress['epoch']+1):
  for stem in ['epoch','journal']:shutil.copyfile(s/f'{stem}{epoch:03}.json',d/f'{stem}{epoch:03}.json')
  j=read(d/f'journal{epoch:03}.json');r=read(d/f'epoch{epoch:03}.json');assert j['epoch']==r['epoch']==epoch and j['previous_journal_sha256']==previous and j['record_sha256']==sha(d/f'epoch{epoch:03}.json') and j['state_sha256']==r['state_sha256'];previous=sha(d/f'journal{epoch:03}.json')
 assert previous==progress['journal_sha256']
 state=e.torch.load(d/'committed_full_state.pt',map_location='cpu',weights_only=True);record=read(d/f'epoch{progress["epoch"]:03}.json');assert state['epoch']==progress['epoch'] and state['updates']==progress['updates'] and e.semantic(state)==record['state_sha256']
 for n in ['split.npz','initial.pt']:shutil.copyfile(s/n,d/n)
 first=f/'run/train'
 for n in ['FIRST_READY.json','FIRST_RELEASE.json','first_update.pt']:shutil.copyfile(first/n,d/n)
 assert read(d/'FIRST_RELEASE.json')['native_first_update_exact'] and sha(d/'first_update.pt')==read(d/'FIRST_RELEASE.json')['first_sha256']
 proof=dict(state='immutable_committed_native_state_full_optimizer_allRNG_journal_and_step_prefix_verified_while_original_worker_runs',held=h,stage=stage,epoch=progress['epoch'],updates=progress['updates'],checkpoint_sha256=progress['checkpoint_sha256'],native_state_sha256=e.semantic(state),step_prefix_sha256=progress['prefix_sha256'],journal_chain_verified=True,all_step_indices_verified=True,native_first_update_exact=True,phase_identity=now,resource_ledger=ledger,frozen_scientific_sha256=sha(P/'FROZEN_HASHES.json'),diagnostic_labels_read=False,users7_8_labels_used=False,complete_fold_training=False,complete_sixfold_cohort=False,automatic_resume_authorized=False,new_fitting_or_inference_started=False)
 write(d/'SNAPSHOT_VERIFIED.json',proof)
 files={'snapshot/'+q.name:q for q in d.iterdir() if q.is_file()}
 for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REAL_RELEASE.json','REMOTE_PREREGISTRATION.json','REUSE_PROOFS.json']:files['phase/'+n]=P/n
 c=read(f/'PROTOCOL.json')
 for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
 for n in ['CLOUD_DOWNLOAD_INDEX.json','MANIFEST.json','READBACK_VERIFIED.json']:files['private_initial/'+n]=P/'initial_backup'/n
 layout=d/'SOURCE_LAYOUT.json';write(layout,dict(archive_to_expected_path={n:str(q) for n,q in files.items()},raw_arrays_external=True,restore_requires_evidence_preserving_recovery_no_auto_relaunch=True));files['SOURCE_LAYOUT.json']=layout
 out.parent.mkdir(parents=True,exist_ok=True);m=build(out,f'mmd_loso6_progress_held{h}_{stage}_20261106',files,'incomplete_committed_training_prefix_only_no_complete_fold_or_sixfold_cohort',chunk_size=7340032,limit=67108864)
 m.update(state='partial_committed_native_checkpoint_and_exact_prefix_streamed',complete_modelset=False,complete_fold_training=False,complete_sixfold_cohort=False,held=h,stage=stage,checkpoint_epoch=progress['epoch'],checkpoint_updates=progress['updates'],snapshot_and_build_wall=time.monotonic()-START,snapshot_and_build_CPU=time.process_time()-CPU_START);write(out/'MANIFEST.json',m);return m

def check(h,stage):
 start=time.monotonic();cpu=time.process_time();d=P/f'progress_backups/held{h}_{stage}';m=read(d/'MANIFEST.json');assert m['complete_modelset'] is False and m['held']==h and m['stage']==stage
 rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
 for r in rows:
  q=r['metadata'];assert q['shared'] is False and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
  assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
 v=verify(m,rows);assert v['complete_modelset_verified'] is False;write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
 write(d/'READBACK_COST.json',dict(wall=time.monotonic()-start,CPU=time.process_time()-cpu,scope='whole_archive_and_every_member_decompression_hash_verification_only'))
 index=dict(state='partial_committed_native_checkpoint_private_owner_only_raw_whole_and_allmembers_verified',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,held=h,stage=stage,checkpoint_epoch=m['checkpoint_epoch'],checkpoint_updates=m['checkpoint_updates'],complete_modelset=False,complete_fold_training=False,complete_sixfold_cohort=False,automatic_resume_authorized=False,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows]);write(d/'CLOUD_DOWNLOAD_INDEX.json',index);return v
if __name__=='__main__':print(json.dumps(create(int(sys.argv[2]),sys.argv[3]) if sys.argv[1]=='build' else check(int(sys.argv[2]),sys.argv[3])))
