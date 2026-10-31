"""Incremental audited2/3 evidence plus immutable first-update4/5 insurance.

Selected/last2/3 models stay in the already verified14-chunk producer archive.
This supplement links those identical bytes to full replay audits. No scores.
"""
from pathlib import Path
import importlib.util,json,shutil,sys,tarfile,time
from verify_and_observe import B,OLD,HERE,read,sha,verify_audits,observe
OUT=HERE/'snapshot01';SCOPE='full_audits23_linked_to_private_models_and_first_updates45'
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
def core():
 assert sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229';spec=importlib.util.spec_from_file_location('preservation_core',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def build():
 a=verify_audits();o=observe();cfg=read(B/'PROTOCOL.json');c=core();OUT.mkdir(exist_ok=False);payload=OUT/'payload';payload.mkdir();members=[];seen=set()
 def copy(source,name):
  source=Path(source);name=str(name);c.relative_name(name);assert name not in seen and source.is_file() and not source.is_symlink();seen.add(name);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,target);assert sha(source)==sha(target)==h;members.append(dict(path=name,bytes=target.stat().st_size,sha256=h))
 def generated(name,value):
  target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2)+'\n');members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=sha(target)))
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','LAUNCH_RECEIPT.json']:copy(B/name,Path('study')/name)
 for held in [2,3]:
  copy(B/'run01'/('held'+str(held))/'COMPLETE_AUDIT.json',Path('audited23')/('held'+str(held))/'COMPLETE_AUDIT.json')
  copy(B/'jobs'/('held'+str(held))/'STATE.json',Path('audited23')/('held'+str(held))/'JOB_STATE.json')
  for kind in ['train','audit']:
   for p in sorted((B/'jobs'/('held'+str(held))/kind).iterdir()):
    if p.is_file():copy(p,Path('audited23')/('held'+str(held))/kind/p.name)
 for name in ['MANIFEST.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json','CLOUD_DOWNLOAD_INDEX.json']:copy(OLD/name,Path('producer23_recovery_reference')/name)
 copy(OLD/'payload/MEMBERS.json',Path('producer23_recovery_reference/MEMBERS.json'))
 for held in [4,5]:
  fold=B/'run01'/('held'+str(held));state=o['folds'][str(held)]['worker'];assert state['state']=='training' and state['total_steps']>=1 and state['diagnostic_labels_read'] is False
  for name in ['IDENTITY.json','split.json','EVAL_BATCH_IDS.json','initial.pt','first_update.pt']:copy(fold/name,Path('first_updates45')/('held'+str(held))/name)
 generated(Path('OBSERVED_LIVE_STATE.json'),o)
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,name in enumerate(cfg['dependencies_text']):copy(name,Path('dependencies/text')/(str(i)+'_'+Path(name).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'));copy(CORE,Path('preservation/complete_modelset.py'))
 for p in sorted(HERE.iterdir()):
  if p.is_file() and p.suffix in ['.py','.md','.json','.log'] and p.name not in ['BUILD.log','VERIFY.log']:copy(p,Path('preservation')/p.name)
 after=verify_audits(write=False);assert after['rows']==a['rows'] and after['producer_files_linked_to_old_private_archive']==a['producer_files_linked_to_old_private_archive']
 metadata=dict(state='immutable_partial_source_confirmation_audited23_first45_supplement',scope=SCOPE,members=members,created_unix=time.time(),fully_audited_new_folds=[2,3],first_update_folds=[4,5],producer23_reference_archive_sha256=a['producer23_archive_sha256'],producer23_reference_receipt_sha256=a['producer23_receipt_sha256'],recovery_requires_existing_verified_producer23_archive=True,linked_producer_files=38,all_five_new_folds_complete=False,complete_source6_modelset=False,all_completed_models_included=False,diagnostic_labels_scored=False,users7_8_labels_used=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,full_job_resume_snapshot=False,limitations='Supplement includes full independent audits/terminations2/3 and initial/first states/IDs/code4/5. To recover audited selected/last2/3 models also fetch the existing verified14-chunk producer23 archive; every38 archived producer file remains byte-identical to later audit evidence. No duplicate selected/last model upload. Folds4/5 not complete, fold6 pending; no source6 accuracy or complete-modelset claim. Original raw/maps/runtime restore separately; no automatic whole-job resume.')
 (payload/'MEMBERS.json').write_text(json.dumps(metadata,indent=2)+'\n');prefix='hinception_source6_audited23_first45_20261031';archive=OUT/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=OUT/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 manifest={k:v for k,v in metadata.items() if k!='members'};manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');c.verify_members(archive,manifest['members_manifest_sha256'],len(members));print(json.dumps(dict(state='local_audited23_first45_supplement_members_verified',bytes=manifest['archive_bytes'],chunks=len(chunks),members=len(members)+1)))
def verify():
 m=read(OUT/'MANIFEST.json');assert m['scope']==SCOPE and m['fully_audited_new_folds']==[2,3] and m['first_update_folds']==[4,5] and m['complete_source6_modelset'] is False;rows=read(OUT/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=OUT/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
   with p.open('rb') as part:shutil.copyfileobj(part,f)
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'];counts=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files']);assert counts['metadata']['scope']==SCOPE and counts['metadata']['complete_source6_modelset'] is False
 proof=dict(state='remote_archive_and_all_members_verified',scope=SCOPE,chunks_verified=len(rows),tracked_payload_members_verified=counts['payload_members'],archive_members=counts['archive_members'],archive_sha256=m['archive_sha256'],fully_audited_new_folds=[2,3],first_update_folds=[4,5],producer23_reference_archive_sha256=m['producer23_reference_archive_sha256'],complete_modelset_verified=False);(OUT/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':assert sys.argv[1:] in [['build'],['verify']];build() if sys.argv[1]=='build' else verify()
