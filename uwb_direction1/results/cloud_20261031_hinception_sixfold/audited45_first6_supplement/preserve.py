"""Incremental audited4/5 evidence plus initial/first6 only; no scoring."""
from pathlib import Path
import json,shutil,sys,tarfile,time
from verify_audits import B,OLD,HERE,TOOLS,read,sha,require,core,verify,observe
OUT=HERE/'snapshot01';SCOPE='full_audits45_linked_to_private_models_and_initial_first6_only'
def build():
 a=verify();o=observe();require(set(o['folds'])=={'6'} and o['folds']['6']['phase']=='training' and o['folds']['6']['worker']['total_steps']>=1 and not o['phase_transitions_or_unverified'],'Final producer6 not actually active');cfg=read(B/'PROTOCOL.json');c=core();OUT.mkdir(exist_ok=False);payload=OUT/'payload';payload.mkdir();members=[];seen=set()
 def copy(source,name):
  source=Path(source);name=str(name);c.relative_name(name);require(name not in seen and source.is_file() and not source.is_symlink(),'Duplicate/nonregular member');seen.add(name);dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,dest);require(sha(source)==sha(dest)==h,'Changed during copy');members.append(dict(path=name,bytes=dest.stat().st_size,sha256=h))
 def generated(name,value):
  p=payload/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');members.append(dict(path=str(name),bytes=p.stat().st_size,sha256=sha(p)))
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','LAUNCH_RECEIPT.json']:copy(B/name,Path('study')/name)
 for held in [4,5]:
  copy(B/'run01'/('held'+str(held))/'COMPLETE_AUDIT.json',Path('audited45')/('held'+str(held))/'COMPLETE_AUDIT.json');copy(B/'jobs'/('held'+str(held))/'STATE.json',Path('audited45')/('held'+str(held))/'JOB_STATE.json')
  for kind in ['train','audit']:
   for p in sorted((B/'jobs'/('held'+str(held))/kind).iterdir()):
    if p.is_file():copy(p,Path('audited45')/('held'+str(held))/kind/p.name)
  for name in ['MANIFEST.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json','CLOUD_DOWNLOAD_INDEX.json']:copy(OLD[held]/name,Path('producer_recovery_references')/('held'+str(held))/name)
  copy(OLD[held]/'payload/MEMBERS.json',Path('producer_recovery_references')/('held'+str(held))/'MEMBERS.json')
 for name in ['IDENTITY.json','split.json','EVAL_BATCH_IDS.json','initial.pt','first_update.pt']:copy(B/'run01/held6'/name,Path('first_update6')/name)
 generated(Path('OBSERVED_LIVE_STATE.json'),o)
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,name in enumerate(cfg['dependencies_text']):copy(name,Path('dependencies/text')/(str(i)+'_'+Path(name).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'));copy('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py',Path('preservation/complete_modelset.py'))
 for name in ['producer_snapshot.py','observe_current.py']:copy(TOOLS/name,Path('preservation')/name)
 for p in sorted(HERE.iterdir()):
  if p.is_file() and p.suffix in ['.py','.md','.json','.log'] and p.name not in ['BUILD.log','VERIFY.log']:copy(p,Path('preservation')/p.name)
 after=verify(write=False);require(after['rows']==a['rows'] and after['linked_producer_files']==a['linked_producer_files'],'Audit/model bindings changed after copy');metadata=dict(state='immutable_partial_audited45_and_initial_first6_supplement',scope=SCOPE,created_unix=time.time(),members=members,fully_audited_folds_in_this_supplement=[4,5],initial_first_only_folds=[6],producer_model_recovery_references=a['model_recovery_references'],requires_existing_producer4_and5_archives=True,linked_producer_files=38,latest_producer6_weights_preserved=False,producer6_preserved_steps=[0,1],complete_source6_modelset=False,all_completed_models_included=False,diagnostic_labels_scored=False,users7_8_labels_used=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,whole_job_resume_snapshot=False,limitations='Full independent audit/termination evidence4/5 links38 unchanged files in existing private7-chunk producer4 and7-chunk producer5 archives, which are required to recover selected/last models. Only immutable initial/first model/optimizer/RNG/IDs for6, not its observed latest epoch weights, selected/last/final or complete trajectory. No duplicate trained model upload, no source6 accuracy, no whole-job resume. Raw/maps/runtime restore separately.')
 (payload/'MEMBERS.json').write_text(json.dumps(metadata,indent=2)+'\n');prefix='hinception_source6_audited45_first6_20261101';archive=OUT/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=OUT/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 m={k:v for k,v in metadata.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(OUT/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');c.verify_members(archive,m['members_manifest_sha256'],len(members));print(json.dumps(dict(state='local_audited45_initial_first6_archive_members_verified',bytes=m['archive_bytes'],chunks=len(chunks),members=len(members)+1)))
def verify_readback():
 m=read(OUT/'MANIFEST.json');require(m['scope']==SCOPE and m['fully_audited_folds_in_this_supplement']==[4,5] and m['initial_first_only_folds']==[6] and m['complete_source6_modelset'] is False,'Wrong/overstated scope');rows=read(OUT/'READBACK_INPUT.json');require(len(rows)==len(m['chunks']),'Wrong part count');archive=OUT/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);require(r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'],'Changed/misordered chunk')
   with p.open('rb') as part:shutil.copyfileobj(part,f)
 require(archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'],'Changed assembled archive');counts=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files']);require(counts['metadata']['scope']==SCOPE and counts['metadata']['producer6_preserved_steps']==[0,1] and counts['metadata']['latest_producer6_weights_preserved'] is False,'Changed embedded preservation scope');proof=dict(state='remote_archive_and_all_members_verified',scope=SCOPE,chunks_verified=len(rows),tracked_payload_members_verified=counts['payload_members'],archive_members=counts['archive_members'],archive_sha256=m['archive_sha256'],fully_audited_folds_in_supplement=[4,5],producer6_preserved_steps=[0,1],complete_modelset_verified=False);(OUT/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':require(sys.argv[1:] in [['build'],['verify']],'build or verify required');build() if sys.argv[1]=='build' else verify_readback()
