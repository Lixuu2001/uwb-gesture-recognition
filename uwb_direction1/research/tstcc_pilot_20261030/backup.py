"""First-update insurance or completed standalone model/evidence preservation."""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from audit_complete import require_complete,read,sha
B=Path(__file__).resolve().parent
def build(out,first=False):
 if first:
  proof=read(B/'FIRST_UPDATE_VERIFIED.json');assert proof['state']=='pass' and proof['first_update_sha256']==sha(B/'run01/ssl/first_update.pt')
 else:
  require_complete(B);audit=read(B/'COMPLETE_AUDIT.json');assert audit['state']=='pass'
  term=read(B/'audit_supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout']
  for p,h in audit['artifact_hashes'].items():assert sha(p)==h,p
 for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  source=Path(source);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,target);assert sha(source)==sha(target)==h;members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=h))
 for p in sorted(B.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.md','.log') and p.name not in ('LAUNCH.log','AUDIT_LAUNCH.log','BACKUP_BUILD.log'):copy(p,Path('study')/p.name)
 run=B/'run01'
 if first:
  for n in ('IDENTITY.json','split.json','AUGMENTATION.json','augmentation_recipe.pt','ssl/initial.pt','ssl/first_update.pt'):copy(run/n,Path('run01')/n)
  (payload/'OBSERVED_LIVE_STATE.json').write_text(json.dumps(read(run/'STATE.json'),indent=2));p=payload/'OBSERVED_LIVE_STATE.json';members.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
 else:
  for p in sorted(run.rglob('*')):
   if p.is_file() and p.name!='worker.lock':copy(p,Path('run01')/p.relative_to(run))
  for folder in ('supervisor01','audit_supervisor01'):
   for p in sorted((B/folder).iterdir()):
    if p.is_file():copy(p,Path(folder)/p.name)
 cfg=read(B/'PROTOCOL.json');screen=Path(cfg['screen'])
 for name in ('author_adapter.py','SOURCE_INDEX.json','EINOPS_DEPENDENCY_INDEX.json','fetch_author_sources.py'):copy(screen/name,Path('dependencies/screen')/name)
 for row in read(screen/'SOURCE_INDEX.json')['files']:copy(screen/'vendor/TS-TCC'/row['path'],Path('dependencies/author')/row['path'])
 deps=read(screen/'EINOPS_DEPENDENCY_INDEX.json')
 for n,h in deps['files'].items():assert sha(Path(deps['base'])/n)==h;copy(Path(deps['base'])/n,Path('dependencies/einops')/n)
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,p in enumerate(cfg['prior_evidence_sha256']):copy(p,Path('dependencies/prior_evidence')/(str(i)+'_'+Path(p).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 scope='first_update_only' if first else 'complete_four_phase_models_and_evidence'
 snapshot=dict(state='immutable_TS_TCC_private_snapshot',scope=scope,observed_unix=time.time(),members=members,raw_arrays_included=False,source_maps_included=False,fixed_augmented_arrays_included=False,fixed_augmentation_recipe_and_hashes_included=True,full_job_resume_snapshot=False,all_completed_models_included=not first,limitations='First snapshot supplies onlySSLinitial/first update and fixed recipe/IDs/code, not live/latest/all phases. Complete snapshot supplies all four phase checkpoints/optimizer/scheduler/RNG/IDs/logs/predictions and audit. Source maps regenerate from private migration with exact RMS; augmented views regenerate from pinned algorithm/recipe. Tested restore helpers exist; no automatic job-resume. No new fit/scoring by backup.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));prefix='tstcc_first_update_20261030' if first else 'tstcc_complete_20261030';archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  for i in range(1000):
   data=f.read(7*1024*1024)
   if not data:break
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)))
 m={k:v for k,v in snapshot.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),tracked_payload_files=len(members),members_manifest_sha256=sha(payload/'MEMBERS.json'),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k!='chunks'}))
def verify(out):
 m=read(out/'MANIFEST.json');rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'];f.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];expected={r['path'] for r in members}|{'MEMBERS.json'}
  assert len(members)==m['tracked_payload_files'] and len(tar.getmembers())==len(expected) and set(tar.getnames())==expected
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 result=dict(state='remote_archive_and_all_members_verified',scope=m['scope'],chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'])
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['first','complete','verify']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.action=='verify':verify(a.out)
 else:build(a.out,first=a.action=='first')
