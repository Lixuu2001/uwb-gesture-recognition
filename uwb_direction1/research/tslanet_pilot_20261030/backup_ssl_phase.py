"""Private immutable SSL-stage archive; no live classifier, raw maps or scoring.

This is a completed-stage preservation bundle, not a full-job resume snapshot.
"""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from audit_complete import require_phase,PHASES,digest,read
BASE=Path(__file__).resolve().parent
def build(out):
 phase,_,_=require_phase(BASE,*PHASES[0]);audit=read(BASE/'SSL_PHASE_AUDIT.json');assert audit['state']=='pass'
 term=read(BASE/'ssl_audit_supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout']
 cfg=read(BASE/'PROTOCOL.json')
 for p,h in audit['artifact_hashes'].items(): assert digest(p)==h,p
 for n,h in read(BASE/'AUDIT_HASHES.json').items(): assert digest(BASE/n)==h,n
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  source=Path(source);dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);before=digest(source);shutil.copyfile(source,dest)
  assert digest(source)==digest(dest)==before,str(source);members.append(dict(path=str(name),bytes=dest.stat().st_size,sha256=before))
 # Explicit immutable metadata only; never capture a classifier checkpoint or live log.
 names=['PROTOCOL.json','PREREGISTRATION.md','FROZEN_HASHES.json','COST_CHECK.json','PREFLIGHT.json','PROTOCOL_TESTS.log','STARTUP_SMOKE_FINAL.log','FIRST_UPDATE_VERIFIED.json','LAUNCH.json','LAUNCH_VERIFIED.md','AUDIT_HASHES.json','AUDIT_TESTS.log','SSL_PHASE_AUDIT.json','SSL_AUDIT_LAUNCH.log']
 names+=sorted(p.name for p in BASE.glob('*.py'))
 for n in names: copy(BASE/n,Path('study')/n)
 for n in ('IDENTITY.json','split.json'): copy(BASE/'run01'/n,Path('run_metadata')/n)
 for p in sorted(phase.iterdir()):
  assert p.is_file() and p.suffix!='.tmp';copy(p,Path('pretrained/pretrain')/p.name)
 for p in sorted((BASE/'ssl_audit_supervisor01').iterdir()):
  if p.is_file(): copy(p,Path('ssl_audit_supervisor01')/p.name)
 prototype=Path(cfg['prototype'])
 for p in sorted(prototype.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pt','.npy','.npz'):
   # Prototype contains only author source, reviewed paper and synthetic artifacts.
   copy(p,Path('dependencies/tslanet')/p.relative_to(prototype))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'));assert digest(payload/'dependencies/source_ids.npz')==cfg['source_manifest_sha256']
 for key in ('previous_state','previous_audit','previous_termination','previous_private_backup','prior_held1_summary'): copy(cfg[key],Path('dependencies')/(key+'.json'))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 snapshot=dict(state='immutable_completed_SSL_phase_snapshot',phase='pretrained/pretrain',epochs=50,selected_epoch=audit['phase']['selected_epoch'],observed_unix=time.time(),members=members,raw_arrays_included=False,source_maps_included=False,live_classifier_included=False,full_job_resume_snapshot=False,real_map_or_label_values_read=False,real_forward_or_fit=False,limitations='All four immutable SSL model snapshots plus optimizer/RNG/IDs/code/history/audit. Classifier trajectory is still running and absent. Restore original data by persistent migration index and regenerate the declared source RMS maps. Original stochastic masked validation loss not independently replayed. No accuracy or novelty evidence.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));archive=out/'tslanet_ssl50_20261030.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file(): tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as stream:
  for i in range(1000):
   data=stream.read(7*1024*1024)
   if not data: break
   p=out/f'tslanet_ssl50_20261030.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=digest(p)))
 manifest={k:v for k,v in snapshot.items() if k!='members'}
 manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),tracked_payload_files=len(members),members_manifest_sha256=digest(payload/'MEMBERS.json'),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='chunks'}))
def verify(out):
 m=read(out/'MANIFEST.json');rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as stream:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert c['name']==r['name'] and p.stat().st_size==c['bytes'] and digest(p)==c['sha256'];stream.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and digest(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];expected={r['path'] for r in members}|{'MEMBERS.json'}
  assert len(members)==m['tracked_payload_files'] and set(tar.getnames())==expected and len(tar.getmembers())==len(expected)
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 result=dict(state='remote_archive_and_all_members_verified',phase='pretrained/pretrain',epochs=50,chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'],full_job_resume_snapshot=False)
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify-readback']);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 (build if args.action=='build' else verify)(args.out)
