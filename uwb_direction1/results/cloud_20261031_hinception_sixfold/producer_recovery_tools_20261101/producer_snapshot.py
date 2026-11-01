"""Guarded per-producer recovery for remaining fixed source folds4/5/6.

Operational only: no model/array/label decode, fits, scoring or job launch.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,shutil,sys,tarfile,time
B=Path('/workspace/uwb-hinception-sixfold-20261031');HERE=Path(__file__).resolve().parent
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def require(value,message):
 if not value:raise RuntimeError(message)
def core():
 require(sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229','Changed pinned member checker');spec=importlib.util.spec_from_file_location('producer_recovery_core',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate(base,cfg,held):
 require(held in [4,5,6],'Only remaining producers4/5/6');fold=base/'run01'/('held'+str(held));s=read(fold/'STATE.json')
 # Before termination/metadata/model reads or destination creation.
 require(s['state']=='complete' and s['pid'] is None and s['all_predictions_frozen'] and s['diagnostic_labels_read'] is False and s['users7_8_labels_used'] is False,'Producer still active/incomplete')
 n=cfg['counts'][str(held)][0];steps=100*((n+63)//64);term=read(base/'jobs'/('held'+str(held))/'train/TERMINATION.json');require(term['exit_code']==0 and term['wall_timeout'] is False,'Unsuccessful training termination');done=read(fold/'TRAINING_COMPLETE.json');result=read(fold/'TRAINING_RESULT.json');require(done['epochs']==result['epochs']==100 and done['steps']==result['steps']==s['total_steps']==steps and result['held']==held and result['training_appearances']==100*n and result['diagnostic_labels_read'] is False and result['users7_8_labels_used'] is False,'Wrong producer identity/counts')
 identity=read(fold/'IDENTITY.json');require(identity['held']==held and identity['protocol_sha256']==sha(base/'PROTOCOL.json'),'Wrong protocol identity');files=read(fold/'FILES.json');required={'initial.pt','first_update.pt','best.pt','last.pt','final_prediction_state.pt','best_model.keras','last_model.keras','split.json','EVAL_BATCH_IDS.json','iterations.jsonl','epochs.jsonl','STATE.json','TRAINING_COMPLETE.json','TRAINING_RESULT.json','IDENTITY.json','ALL_PREDICTIONS_FROZEN.json','validation_predictions_unscored.npz','diagnostic_predictions_unscored.npz'};require(required<=set(files),'Missing immutable producer artifact')
 for name,h in files.items():require(sha(fold/name)==h,'Changed producer artifact: '+name)
 for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]:require(sha(fold/name)==done[key],'Changed checkpoint binding')
 frozen=read(fold/'ALL_PREDICTIONS_FROZEN.json');require(set(frozen)=={'validation','diagnostic'},'Incomplete prediction freeze')
 for role,h in frozen.items():require(sha(fold/(role+'_predictions_unscored.npz'))==h,'Changed prediction freeze')
 return files
def build(held,out):
 sys.path.insert(0,str(B));from readiness import ready
 cfg=read(B/'PROTOCOL.json');files=validate(B,cfg,held);ready(cfg);c=core();out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];seen=set()
 def copy(source,name):
  source=Path(source);name=str(name);c.relative_name(name);require(name not in seen and source.is_file() and not source.is_symlink(),'Duplicate/nonregular member');seen.add(name);dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,dest);require(sha(source)==sha(dest)==h,'Changed during copy');members.append(dict(path=name,bytes=dest.stat().st_size,sha256=h))
 def generated(name,value):
  p=payload/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');members.append(dict(path=str(name),bytes=p.stat().st_size,sha256=sha(p)))
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','LAUNCH_RECEIPT.json']:copy(B/name,Path('study')/name)
 fold=B/'run01'/('held'+str(held))
 for name in sorted(files):
  if name!='worker.lock':copy(fold/name,Path('run01')/('held'+str(held))/name)
 copy(fold/'FILES.json',Path('run01')/('held'+str(held))/'FILES.json')
 for p in sorted((B/'jobs'/('held'+str(held))/'train').iterdir()):
  if p.is_file():copy(p,Path('jobs')/('held'+str(held))/'train'/p.name)
 generated(Path('OBSERVED_QUEUE_STATE.json'),read(B/'STATE.json'));generated(Path('OBSERVED_JOB_STATE.json'),read(B/'jobs'/('held'+str(held))/'STATE.json'))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,name in enumerate(cfg['dependencies_text']):copy(name,Path('dependencies/text')/(str(i)+'_'+Path(name).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'));copy(CORE,Path('preservation/complete_modelset.py'))
 for p in sorted(HERE.iterdir()):
  if p.is_file() and p.suffix in ['.py','.json','.md','.log'] and p.name not in ['BUILD.log','VERIFY.log']:copy(p,Path('preservation')/p.name)
 require(validate(B,cfg,held)==files,'Producer changed after copy');n=cfg['counts'][str(held)][0];scope=f'held{held}_training_complete_producer_recovery_no_numeric_audit_claim';metadata=dict(state='immutable_single_completed_source_producer',scope=scope,held=held,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),members=members,epochs=100,updates=100*((n+63)//64),training_appearances=100*n,all_selected_last_initial_first_final_native_states_included=True,numeric_audit_certified_by_this_snapshot=False,complete_source6_modelset=False,all_completed_models_included=False,diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,full_job_resume_snapshot=False,limitations='Complete immutable artifacts for this single successful producer only; does not certify its full numerical replay, source6 accuracy, other producer models or whole-job resume. Exact IDs/logs/predictions/models remain private. Raw/maps/runtime restore separately. Later full audited six-model preservation is still required.')
 (payload/'MEMBERS.json').write_text(json.dumps(metadata,indent=2)+'\n');prefix=f'hinception_source6_producer{held}_20261101';archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 manifest={k:v for k,v in metadata.items() if k!='members'};manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');c.verify_members(archive,manifest['members_manifest_sha256'],len(members));print(json.dumps(dict(state='single_producer_archive_local_members_verified',held=held,bytes=manifest['archive_bytes'],chunks=len(chunks),members=len(members)+1)))
def verify(held,out):
 m=read(out/'MANIFEST.json');require(m['held']==held and m['scope']==f'held{held}_training_complete_producer_recovery_no_numeric_audit_claim' and m['complete_source6_modelset'] is False,'Wrong/overstated scope');rows=read(out/'READBACK_INPUT.json');require(len(rows)==len(m['chunks']),'Wrong chunk count');archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);require(r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'],'Changed/misordered readback part')
   with p.open('rb') as part:shutil.copyfileobj(part,f)
 require(archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'],'Changed assembled archive');counts=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files']);require(counts['metadata']['held']==held and counts['metadata']['scope']==m['scope'] and counts['metadata']['complete_source6_modelset'] is False,'Changed embedded scope');proof=dict(state='remote_archive_and_all_members_verified',scope=m['scope'],held=held,chunks_verified=len(rows),tracked_payload_members_verified=counts['payload_members'],archive_members=counts['archive_members'],archive_sha256=m['archive_sha256'],numeric_audit_certified_by_this_snapshot=False,complete_modelset_verified=False);(out/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify']);p.add_argument('--held',type=int,choices=[4,5,6],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.held,a.out) if a.action=='build' else verify(a.held,a.out)
