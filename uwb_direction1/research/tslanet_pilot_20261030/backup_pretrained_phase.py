"""Completed pretrained classifier + opaque predictions; no live scratch state.

Recovery requires the separately verified SSL50 bundle for SSL model snapshots.
No diagnostic array decoding or scoring; this is not a full-job resume snapshot.
"""
import argparse,json,shutil,tarfile,time,hashlib
from pathlib import Path
from audit_pretrained_phase import require_ready,digest,read
BASE=Path(__file__).resolve().parent
def build(out):
 done=require_ready(BASE);audit=read(BASE/'PRETRAINED_PHASE_AUDIT.json');assert audit['state']=='pass'
 term=read(BASE/'pretrained_audit_supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout']
 old=read(BASE/'ssl_backup_20261030/DRIVE_BACKUP_RECEIPT.json');assert old['state']=='private_readback_verified' and old['phase']=='pretrained/pretrain'
 for p,h in audit['artifact_hashes'].items():assert digest(p)==h,p
 for index in ('AUDIT_HASHES.json','PRETRAINED_PHASE_AUDIT_HASHES.json'):
  for n,h in read(BASE/index).items():assert digest(BASE/n)==h,n
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  source=Path(source);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);before=digest(source);shutil.copyfile(source,target)
  assert digest(source)==digest(target)==before,str(source);members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=before))
 names=['PROTOCOL.json','PREREGISTRATION.md','FROZEN_HASHES.json','COST_CHECK.json','PREFLIGHT.json','PROTOCOL_TESTS.log','STARTUP_SMOKE_FINAL.log','FIRST_UPDATE_VERIFIED.json','LAUNCH.json','LAUNCH_VERIFIED.md','AUDIT_HASHES.json','AUDIT_TESTS.log','SSL_PHASE_AUDIT.json','PRETRAINED_PHASE_AUDIT_HASHES.json','PRETRAINED_PHASE_AUDIT_TESTS.log','PRETRAINED_PHASE_AUDIT.json','PRETRAINED_PHASE_AUDIT_LAUNCH.log']
 names+=sorted(p.name for p in BASE.glob('*.py'))
 for n in names:copy(BASE/n,Path('study')/n)
 for n in ('IDENTITY.json','split.json'):copy(BASE/'run01'/n,Path('run_metadata')/n)
 for p in sorted((BASE/'run01/pretrained/classification').iterdir()):
  assert p.is_file() and p.suffix!='.tmp';copy(p,Path('pretrained/classification')/p.name)
 for n in ('ARM_COMPLETE.json','predictions_unscored.npz'):copy(BASE/'run01/pretrained'/n,Path('pretrained')/n)
 for p in sorted((BASE/'pretrained_audit_supervisor01').iterdir()):
  if p.is_file():copy(p,Path('pretrained_audit_supervisor01')/p.name)
 for n in ('DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json'):copy(BASE/'ssl_backup_20261030'/n,Path('dependencies/SSL50')/n)
 cfg=read(BASE/'PROTOCOL.json');prototype=Path(cfg['prototype'])
 for p in sorted(prototype.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pt','.npy','.npz'):copy(p,Path('dependencies/tslanet')/p.relative_to(prototype))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'));assert digest(payload/'dependencies/source_ids.npz')==cfg['source_manifest_sha256']
 for key in ('previous_state','previous_audit','previous_termination','previous_private_backup','prior_held1_summary'):copy(cfg[key],Path('dependencies')/(key+'.json'))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 snapshot=dict(state='immutable_completed_pretrained_classifier_snapshot',arm='pretrained',phase='classification',epochs=100,selected_epoch=audit['phase']['selected_epoch'],observed_unix=time.time(),members=members,current_archive_model_phases=['pretrained/classification'],SSL_models_require_prior_verified_bundle=True,SSL_archive_sha256=old['archive_sha256'],SSL_receipt_sha256=digest(BASE/'ssl_backup_20261030/DRIVE_BACKUP_RECEIPT.json'),raw_arrays_included=False,source_maps_included=False,live_scratch_included=False,full_job_resume_snapshot=False,diagnostic_values_decoded=False,diagnostic_labels_read=False,real_fitting_or_new_selection=False,limitations='Four pretrained classifier snapshots plus optimizer/RNG/IDs, full100epoch log, opaque held1 predictions, source/licenses/audits and priorSSL recovery index. The priorSSL17chunk bundle supplies50epoch SSL snapshots. Live scratch and final paired scores absent. Original data migration and exact RMS regenerates maps. Not full-job resume or final accuracy evidence.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));archive=out/'tslanet_pretrained100_20261030.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as stream:
  for i in range(1000):
   data=stream.read(7*1024*1024)
   if not data:break
   p=out/f'tslanet_pretrained100_20261030.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=digest(p)))
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
 result=dict(state='remote_archive_and_all_members_verified',arm='pretrained',phase='classification',epochs=100,chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'],SSL_models_require_prior_verified_bundle=True,full_job_resume_snapshot=False)
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify-readback']);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 (build if args.action=='build' else verify)(args.out)
