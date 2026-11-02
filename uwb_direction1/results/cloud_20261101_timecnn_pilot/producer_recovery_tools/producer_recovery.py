"""Preserve completed immutable TimeCNN training while original full audit runs.

Operational outside frozen study; no array/model/label decode or job launch.
This producer scope cannot establish complete audited recovery or permit score.
"""
from pathlib import Path
import argparse,datetime,importlib.util,json,shutil,sys,tarfile
B=Path('/workspace/uwb-timecnn-pilot-20261101');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from guards import ready,require_training_complete,read,sha
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
SCOPE='completed_full2000_training_producer_numeric_audit_pending_not_certified'
def core():
 assert sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229'
 spec=importlib.util.spec_from_file_location('TimeCNN_producer_members',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate():
 term=require_training_complete(B);c=ready();run=B/'run01';r=read(run/'TRAINING_RESULT.json');s=read(run/'STATE.json');done=read(run/'TRAINING_COMPLETE.json')
 assert r['epochs']==s['epoch']==done['epochs']==2000 and r['steps']==s['total_steps']==done['steps']==588000 and r['training_appearances']==9400000
 assert not r['diagnostic_labels_read'] and not r['users7_8_labels_used'] and s['all_predictions_frozen'] and s['pid'] is None
 assert read(run/'IDENTITY.json')['protocol_sha256']==sha(B/'PROTOCOL.json')
 files=read(run/'FILES.json');required={'initial.pt','first_update.pt','best.pt','last.pt','final_prediction_state.pt','best_model.keras','last_model.keras','IDENTITY.json','split.json','EVAL_BATCH_IDS.json','iterations.jsonl','epochs.jsonl','TRAINING_COMPLETE.json','TRAINING_RESULT.json','STATE.json','ALL_PREDICTIONS_FROZEN.json','validation_predictions_unscored.npz','diagnostic_predictions_unscored.npz'};assert required<=set(files)
 for name,d in files.items():assert sha(run/name)==d,name
 return c,files,done,term
def build(out):
 c,files,done,term=validate();cc=core();out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];seen=set()
 def copy(source,name):
  source=Path(source);name=str(name);cc.relative_name(name);assert source.is_file() and not source.is_symlink() and name not in seen;seen.add(name);digest=sha(source);dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(source)==sha(dest)==digest;members.append({'path':name,'bytes':dest.stat().st_size,'sha256':digest})
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json']:copy(B/name,Path('study')/name)
 copy(Path('/workspace/uwb-timecnn-monitor-20261101/LAUNCH_RECEIPT.json'),'study/LAUNCH_RECEIPT.json')
 for name in sorted(files):copy(B/'run01'/name,Path('run01')/name)
 copy(B/'run01/FILES.json','run01/FILES.json')
 for p in sorted((B/'supervisor01').iterdir()):
  if p.is_file():copy(p,Path('supervisor01')/p.name)
 for i,path in enumerate(c['dependencies_text']):copy(path,Path('dependencies/text')/(str(i)+'_'+Path(path).name))
 copy(c['source_manifest'],'dependencies/source_ids.npz');source=Path('/workspace/uwb-timecnn-source-probe-20261101')
 for row in read(source/'SOURCE_INDEX.json')['files']:copy(source/'local_author_sources'/row['path'],Path('dependencies/private_author_source')/row['path'])
 copy(CORE,'preservation/complete_modelset.py');copy(Path(__file__),'preservation/producer_recovery.py')
 assert validate()[1]==files
 meta={'state':'immutable_completed_full_TimeCNN_training_producer','scope':SCOPE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epochs':2000,'updates':588000,'training_appearances':9400000,'selected_epoch':done['best_epoch'],'all_produced_initial_first_best_last_final_and_native_checkpoints_included':True,'numeric_audit_certified_by_this_snapshot':False,'complete_modelset_verified':False,'diagnostic_labels_scored':False,'raw_arrays_included':False,'source_maps_included':False,'runtime_binaries_included':False,'whole_job_resume':False,'limitations':'Complete immutable training producer bytes/IDs/logs/unscored predictions and successful train termination only. Independent full numerical audit runs separately and is not certified by this archive. Not a complete audited modelset, score authorization, accuracy or originality result. Raw/maps/runtime restore separately.','members':members}
 (payload/'MEMBERS.json').write_text(json.dumps(meta,indent=2)+'\n');prefix='timecnn2000_completed_producer_20261101';archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append({'name':p.name,'bytes':len(data),'sha256':sha(p)});i+=1
 m={k:v for k,v in meta.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');cc.verify_members(archive,m['members_manifest_sha256'],len(members));print(json.dumps({'state':'completed_training_producer_archive_local_members_verified','bytes':m['archive_bytes'],'chunks':len(chunks),'members':len(members)+1,'scope':SCOPE}))
def verify(out):
 m=read(out/'MANIFEST.json');assert m['scope']==SCOPE and not m['numeric_audit_certified_by_this_snapshot'] and not m['complete_modelset_verified'];rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
   with p.open('rb') as g:shutil.copyfileobj(g,f)
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'];counts=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files']);assert counts['metadata']['scope']==SCOPE and not counts['metadata']['complete_modelset_verified']
 proof={'state':'remote_archive_and_all_members_verified','scope':SCOPE,'chunks_verified':len(rows),'tracked_payload_members_verified':counts['payload_members'],'archive_members':counts['archive_members'],'archive_sha256':m['archive_sha256'],'producer_training_bytes_recoverable':True,'numeric_audit_certified':False,'complete_modelset_verified':False};(out/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify']);p.add_argument('--out',required=True,type=Path);a=p.parse_args();build(a.out) if a.action=='build' else verify(a.out)
