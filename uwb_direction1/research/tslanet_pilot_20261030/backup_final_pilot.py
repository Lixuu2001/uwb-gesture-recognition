"""Preserve finished scratch/final evidence; prior two bundles supply other models."""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from audit_complete import require_complete,read,digest
B=Path(__file__).resolve().parent
def build(out):
 term=require_complete(B);audit=read(B/'COMPLETE_AUDIT.json');assert audit['state']=='pass'
 at=read(B/'audit_supervisor01/TERMINATION.json');assert at['exit_code']==0 and not at['wall_timeout']
 for p,h in audit['artifact_hashes'].items():assert digest(p)==h,p
 prior_paths=[B/'ssl_backup_20261030/DRIVE_BACKUP_RECEIPT.json',B/'pretrained_backup_20261030/DRIVE_BACKUP_RECEIPT.json'];priors=[read(p) for p in prior_paths];assert all(r['state']=='private_readback_verified' for r in priors)
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  source=Path(source);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);h=digest(source);shutil.copyfile(source,target);assert digest(source)==digest(target)==h;members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=h))
 # Root code and immutable reports, excluding any live logging controller.
 for p in sorted(B.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.md','.log') and p.name!='launch.log':copy(p,Path('study')/p.name)
 run=B/'run01'
 for n in ('IDENTITY.json','split.json','STATE.json','SUMMARY.json','FILES.json','ALL_PREDICTIONS_FROZEN.json'):copy(run/n,Path('run_metadata')/n)
 for arm in ('pretrained','scratch'):
  for n in ('ARM_COMPLETE.json','predictions_unscored.npz'):copy(run/arm/n,Path(arm)/n)
  for p in sorted((run/arm/'classification').iterdir()):
   if arm=='scratch' or p.suffix!='.pt':copy(p,Path(arm)/'classification'/p.name)
 for n in ('PHASE_COMPLETE.json','epochs.jsonl'):copy(run/'pretrained/pretrain'/n,Path('pretrained/pretrain')/n)
 for folder in ('supervisor01','audit_supervisor01'):
  for p in sorted((B/folder).iterdir()):
   if p.is_file():copy(p,Path(folder)/p.name)
 for stage,p in zip(('SSL50','pretrained100'),prior_paths):copy(p,Path('dependencies/prior_bundles')/stage/'DRIVE_BACKUP_RECEIPT.json')
 cfg=read(B/'PROTOCOL.json');copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for key in ('previous_state','previous_audit','previous_termination','previous_private_backup','prior_held1_summary'):copy(cfg[key],Path('dependencies')/(key+'.json'))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 snapshot=dict(state='completed_TSLANet_final_evidence_snapshot',observed_unix=time.time(),members=members,current_archive_model_phases=['scratch/classification'],requires_prior_model_bundles=['pretrained/pretrain','pretrained/classification'],prior_archives=[dict(archive_sha256=r['archive_sha256'],receipt_sha256=digest(p)) for p,r in zip(prior_paths,priors)],full_experiment_modelset_recoverable_composite=True,raw_arrays_included=False,source_maps_included=False,full_job_resume_snapshot=False,diagnostic_values_decoded_by_backup=False,limitations='Current archive supplies scratch initial/first/best/last tensors and full final evidence. Prior two verified17chunk bundles supply SSL and pretrained classifier tensors plus author source/licenses. All three recovery indices form a complete modelset composite. Original migration restores raw data; exact declared RMS regenerates source maps. No new fitting/scoring by backup.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));archive=out/'tslanet_final_20261030.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  for i in range(1000):
   data=f.read(7*1024*1024)
   if not data:break
   p=out/f'tslanet_final_20261030.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=digest(p)))
 m={k:v for k,v in snapshot.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),tracked_payload_files=len(members),members_manifest_sha256=digest(payload/'MEMBERS.json'),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k!='chunks'}))
def verify(out):
 m=read(out/'MANIFEST.json');rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and digest(p)==c['sha256'];f.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and digest(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];expected={r['path'] for r in members}|{'MEMBERS.json'}
  assert len(members)==m['tracked_payload_files'] and len(tar.getmembers())==len(expected) and set(tar.getnames())==expected
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 result=dict(state='remote_archive_and_all_members_verified',chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'],full_experiment_modelset_recoverable_composite=True)
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify-readback']);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();(build if args.action=='build' else verify)(args.out)
