"""Private first-state insurance or complete audited modelset; no array decode."""
from pathlib import Path
import argparse,importlib.util,json,shutil,tarfile,time
from guards import ready,read,sha,require_training_complete
B=Path(__file__).resolve().parent
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
def core():
 assert sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229'
 spec=importlib.util.spec_from_file_location('TCN_member_checker',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def complete():
 require_training_complete();c=ready();a=read(B/'COMPLETE_AUDIT.json');t=read(B/'audit_supervisor01/TERMINATION.json');s=read(B/'STATE.json')
 assert a['state']=='pass' and a['updates_replayed']==1480 and a['epochs_replayed']==20 and not a['diagnostic_labels_read'] and not a['users7_8_labels_used']
 assert t['exit_code']==0 and not t['wall_timeout'] and s['state']=='complete_audited_unscored_requires_private_backup_and_finalization' and s['pid'] is None
 for p,d in a['artifact_hashes'].items():assert sha(p)==d,p
 return c
def build(scope,out):
 c=complete() if scope=='complete' else ready();cc=core();run=B/'run01'
 assert (run/'initial.pt').is_file() and (run/'first_update.pt').is_file()
 assert read(run/'IDENTITY.json')['protocol_sha256']==sha(B/'PROTOCOL.json')
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];seen=set()
 def copy(src,name):
  src=Path(src);name=str(name);cc.relative_name(name);assert name not in seen and src.is_file() and not src.is_symlink();seen.add(name);h=sha(src);dst=payload/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst)==h;members.append({'path':name,'bytes':dst.stat().st_size,'sha256':h})
 for name in read(B/'FROZEN_HASHES.json'):
  if not Path(name).is_absolute():copy(B/name,Path('study')/name)
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json']:copy(B/name,Path('study')/name)
 if scope=='complete':
  for p in sorted(run.rglob('*')):
   if p.is_file() and p.name!='worker.lock' and p.suffix!='.tmp':copy(p,Path('run01')/p.relative_to(run))
  for name in ['COMPLETE_AUDIT.json','STATE.json']:copy(B/name,name)
  for group in ['supervisor01','audit_supervisor01']:
   for p in sorted((B/group).iterdir()):
    if p.is_file():copy(p,Path(group)/p.name)
 else:
  for name in ['initial.pt','first_update.pt','IDENTITY.json','split.json','EVAL_BATCH_IDS.json']:copy(run/name,Path('run01')/name)
 for i,path in enumerate(c['dependencies_text']):copy(path,Path('dependencies/text')/(str(i)+'_'+Path(path).name))
 copy(c['source_manifest'],'dependencies/source_ids.npz');copy(CORE,'preservation/complete_modelset.py')
 # Original author sources remain private and may be restored from this bundle.
 src=Path('/workspace/uwb-tcn-source-probe-20261102')
 for r in read(src/'SOURCE_INDEX.json')['files']:copy(src/'local_author_sources'/r['path'],Path('dependencies/private_author_source')/r['path'])
 meta={'state':'immutable_TCN_modelset_snapshot','scope':'complete_audited_full20_modelset' if scope=='complete' else 'initial_first_only_not_latest_or_complete','created_unix':time.time(),'members':members,'complete_modelset':scope=='complete','all_initial_first_epoch10_last_final_and_native_models_included':scope=='complete','full_numerical_audit_included':scope=='complete','diagnostic_labels_scored':False,'raw_arrays_included':False,'source_maps_included':False,'runtime_binaries_included':False,'whole_job_resume':False,'limitations':'Initial/first scope is immutable start insurance only. Complete scope retains all produced checkpoints, exact IDs/logs/predictions and full audit; raw/maps/runtime restore separately. Remote preservation is established only after ordered raw chunk and every-member readback.'}
 (payload/'MEMBERS.json').write_text(json.dumps(meta,indent=2)+'\n');prefix='tcn20_'+scope+'_20261102';archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append({'name':p.name,'bytes':len(data),'sha256':sha(p)});i+=1
 m={k:v for k,v in meta.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');cc.verify_members(archive,m['members_manifest_sha256'],len(members));print(json.dumps({'state':'local_archive_verified','scope':m['scope'],'bytes':m['archive_bytes'],'chunks':len(chunks),'members':len(members)+1}))
def verify(out):
 m=read(out/'MANIFEST.json');rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
   with p.open('rb') as g:shutil.copyfileobj(g,f)
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'];v=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files']);assert v['metadata']['scope']==m['scope'] and v['metadata']['complete_modelset']==m['complete_modelset']
 proof={'state':'remote_archive_and_all_members_verified','scope':m['scope'],'chunks_verified':len(rows),'archive_members':v['archive_members'],'tracked_payload_members_verified':v['payload_members'],'archive_sha256':m['archive_sha256'],'complete_modelset_verified':m['complete_modelset']};(out/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['initial_first','complete','verify']);p.add_argument('--out',required=True,type=Path);a=p.parse_args();verify(a.out) if a.action=='verify' else build('complete' if a.action=='complete' else 'initial_first',a.out)
