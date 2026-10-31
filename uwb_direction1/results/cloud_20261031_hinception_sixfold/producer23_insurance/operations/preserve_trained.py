"""Completed producer folds2/3 insurance, explicitly before full-audit completion.

No model/array/label loading, fits or scoring. No frozen training code changes.
"""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys,tarfile,time
B=Path('/workspace/uwb-hinception-sixfold-20261031')
OUT=Path(__file__).resolve().parent/'snapshot01'
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
SCOPE='held2_held3_training_complete_full_audits_pending'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def core():
 assert sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229'
 spec=importlib.util.spec_from_file_location('complete_preservation_core',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def producer(held,cfg):
 assert held in [2,3];fold=B/'run01'/('held'+str(held));s=read(fold/'STATE.json');term=read(B/'jobs'/('held'+str(held))/'train/TERMINATION.json');n=cfg['counts'][str(held)][0];steps=100*((n+63)//64)
 assert s['state']=='complete' and s['pid'] is None and s['all_predictions_frozen'] and s['diagnostic_labels_read'] is False and s['users7_8_labels_used'] is False and s['total_steps']==steps
 assert term['exit_code']==0 and term['wall_timeout'] is False
 done=read(fold/'TRAINING_COMPLETE.json');result=read(fold/'TRAINING_RESULT.json');assert done['epochs']==result['epochs']==100 and done['steps']==result['steps']==steps and result['training_appearances']==100*n and result['diagnostic_labels_read'] is False
 files=read(fold/'FILES.json')
 required={'initial.pt','first_update.pt','best.pt','last.pt','final_prediction_state.pt','best_model.keras','last_model.keras','split.json','EVAL_BATCH_IDS.json','iterations.jsonl','epochs.jsonl','ALL_PREDICTIONS_FROZEN.json','STATE.json','TRAINING_COMPLETE.json','TRAINING_RESULT.json','IDENTITY.json','validation_predictions_unscored.npz','diagnostic_predictions_unscored.npz'}
 assert required<=set(files)
 for name,h in files.items():assert sha(fold/name)==h,name
 for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]:assert sha(fold/name)==done[key]
 freeze=read(fold/'ALL_PREDICTIONS_FROZEN.json');assert set(freeze)=={'validation','diagnostic'}
 for role,h in freeze.items():assert sha(fold/(role+'_predictions_unscored.npz'))==h
 return files
def build():
 sys.path.insert(0,str(B));from readiness import ready
 cfg=read(B/'PROTOCOL.json');ready(cfg);c=core();checks={h:producer(h,cfg) for h in [2,3]};OUT.mkdir(exist_ok=False);payload=OUT/'payload';payload.mkdir();members=[];seen=set()
 def copy(source,name):
  source=Path(source);name=str(name);c.relative_name(name);assert name not in seen and source.is_file() and not source.is_symlink();seen.add(name);dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,dest);assert sha(source)==sha(dest)==h;members.append(dict(path=name,bytes=dest.stat().st_size,sha256=h))
 def generated(name,value):
  p=payload/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');members.append(dict(path=str(name),bytes=p.stat().st_size,sha256=sha(p)))
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','LAUNCH_RECEIPT.json','RUNTIME_REMOTE_PUBLICATION.json']:copy(B/name,Path('study')/name)
 for held,files in checks.items():
  fold=B/'run01'/('held'+str(held))
  for name in sorted(files):
   if name!='worker.lock':copy(fold/name,Path('run01')/('held'+str(held))/name)
  copy(fold/'FILES.json',Path('run01')/('held'+str(held))/'FILES.json')
  for p in sorted((B/'jobs'/('held'+str(held))/'train').iterdir()):
   if p.is_file():copy(p,Path('jobs')/('held'+str(held))/'train'/p.name)
 generated(Path('OBSERVED_QUEUE_STATE.json'),read(B/'STATE.json'));generated(Path('OBSERVED_JOB_STATES.json'),{str(h):read(B/'jobs'/('held'+str(h))/'STATE.json') for h in [2,3]})
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,name in enumerate(cfg['dependencies_text']):copy(name,Path('dependencies/text')/(str(i)+'_'+Path(name).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 up=Path(cfg['upstream_workspace'])
 for name in ['COMPLETE_AUDIT.json','complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json','complete_backup_20261031/READBACK_VERIFIED.json']:copy(up/name,Path('reused_held1')/name)
 copy('/workspace/uwb-hinception-final-publication-20261031/uwb_direction1/results/cloud_20261031_hinception_pilot/complete_backup/CLOUD_DOWNLOAD_INDEX.json',Path('reused_held1/CLOUD_DOWNLOAD_INDEX.json'))
 copy(CORE,Path('preservation/complete_modelset.py'))
 for p in Path(__file__).resolve().parent.iterdir():
  if p.is_file() and p.suffix in ['.py','.md','.json','.log']:copy(p,Path('preservation')/p.name)
 for held in [2,3]:assert checks[held]==producer(held,cfg)
 snapshot=dict(state='immutable_completed_producer23_audit_pending_insurance',scope=SCOPE,created_unix=time.time(),members=members,producer_folds=[2,3],training_steps={'2':7400,'3':7100},training_epochs_each=100,selected_last_initial_first_final_and_native_models_included=True,all_completed_models_included=False,complete_source_sixfold_modelset=False,full_replay_audits_complete=False,diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,full_environment_or_job_resume_snapshot=False,limitations='All immutable completed producer files/models/ID/log/predictions forheld2/3 and successful training terminations; full audits actively pending, held4/5/6 not yet run. Reused held1 complete modelset referenced by existing private index. This is not complete source6 recovery, full numeric audit or whole-job resume. No accuracy selected/reported by preservation.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2)+'\n');prefix='hinception_source6_trained23_20261031';archive=OUT/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=OUT/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 manifest={k:v for k,v in snapshot.items() if k!='members'};manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');c.verify_members(archive,manifest['members_manifest_sha256'],len(members));print(json.dumps(dict(state='local_producer23_models_archived_no_full_audit_claim',bytes=manifest['archive_bytes'],chunks=len(chunks),members=len(members)+1)))
def verify():
 manifest=read(OUT/'MANIFEST.json');assert manifest['scope']==SCOPE and manifest['full_replay_audits_complete'] is False and manifest['complete_source_sixfold_modelset'] is False;rows=read(OUT/'READBACK_INPUT.json');assert len(rows)==len(manifest['chunks']);archive=OUT/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(manifest['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
   with p.open('rb') as part:shutil.copyfileobj(part,f)
 assert archive.stat().st_size==manifest['archive_bytes'] and sha(archive)==manifest['archive_sha256'];counts=core().verify_members(archive,manifest['members_manifest_sha256'],manifest['tracked_payload_files']);assert counts['metadata']['scope']==SCOPE and counts['metadata']['full_replay_audits_complete'] is False and counts['metadata']['complete_source_sixfold_modelset'] is False
 proof=dict(state='remote_archive_and_all_members_verified',scope=SCOPE,chunks_verified=len(rows),tracked_payload_members_verified=counts['payload_members'],archive_members=counts['archive_members'],archive_sha256=manifest['archive_sha256'],producer_fold_models_preserved=[2,3],full_replay_audits_complete=False,complete_modelset_verified=False);(OUT/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':assert sys.argv[1:] in [['build'],['verify']];build() if sys.argv[1]=='build' else verify()
