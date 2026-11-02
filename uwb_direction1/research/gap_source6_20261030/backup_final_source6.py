"""Completed GAP6 recovery set: held6 tensors + final artifacts + prior indices."""
import argparse,json,shutil,tarfile,time
from pathlib import Path
from backup_completed_fold import sha
BASE=Path(__file__).resolve().parent
def build(out):
 run=BASE/'run01';state=json.loads((run/'STATE.json').read_text())
 assert state['state']=='complete' and state['pid'] is None and set(state['completed_folds'])==set(range(1,7))
 assert state['all_new_diagnostic_predictions_frozen']
 audit=json.loads((BASE/'COMPLETE_AUDIT.json').read_text());assert audit['state']=='pass'
 for folder in ('supervisor01','audit_supervisor01'):
  term=json.loads((BASE/folder/'TERMINATION.json').read_text());assert term['exit_code']==0 and not term['wall_timeout']
 for n,h in audit['artifact_hashes'].items():assert sha(Path(n))==h,n
 for index in ('FROZEN_HASHES.json','AUDIT_HASHES.json'):
  for n,h in json.loads((BASE/index).read_text()).items():assert sha(BASE/n)==h,n
 cfg=json.loads((BASE/'PROTOCOL.json').read_text());out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);before=sha(source);shutil.copyfile(source,target);assert sha(source)==sha(target)==before,str(source)
  members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=before))
 for p in sorted(BASE.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.md','.log'):copy(p,Path('study')/p.name)
 for name in ('STATE.json','SUMMARY.json','IDENTITY.json','REUSED_HELD1.json','ALL_NEW_PREDICTIONS_FROZEN.json','FILES.json'):copy(run/name,Path('run_metadata')/name)
 for held in range(2,7):
  for p in sorted((run/f'held{held}').iterdir()):
   assert p.is_file() and p.suffix!='.tmp'
   if held==6 or p.suffix!='.pt':copy(p,Path(f'held{held}')/p.name)
 for folder in ('supervisor01','audit_supervisor01'):
  for name in ('TERMINATION.json','process.log'):copy(BASE/folder/name,Path(folder)/name)
 for held in range(2,6):
  p=BASE/f'backup_held{held}_20261030/DRIVE_BACKUP_RECEIPT.json';r=json.loads(p.read_text());assert r['state']=='private_readback_verified' and r['manifest']['held']==held
  copy(p,Path('prior_backups')/f'held{held}.json')
 copy(Path(cfg['upstream_private_backup']),Path('prior_backups/upstream_held1_author_core.json'))
 copy(Path(cfg['source_manifest']),Path('dependencies/source_ids.npz'))
 for held,ref in cfg['cnn_references'].items():assert sha(ref['path'])==ref['sha256'];copy(Path(ref['path']),Path('dependencies/cnn_predictions')/f'held{held}.npz')
 copy(Path(cfg['localcnn_audit']),Path('dependencies/prior_localcnn_audit.json'))
 description=dict(state='completed_gap_source6_recoverable_composite',observed_unix=time.time(),members=members,current_archive_models=[6],prior_model_indices={str(h):f'prior_backups/held{h}.json' for h in range(2,6)},held1='prior_backups/upstream_held1_author_core.json',full_set_requires_prior_archives=True,raw_radar_arrays_included=False,source_maps_included=False,scope='All trajectories finished; current archive contains held6 tensors/final metadata and references earlier private model bundles. Never claim this archive alone contains every model. Original input migration index and exact source IDs regenerate signed-map per-record/view RMS/reshape cache.')
 (payload/'MEMBERS.json').write_text(json.dumps(description,indent=2));archive=out/'gap_source6_final.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as stream:
  for i in range(1000):
   data=stream.read(7*1024*1024)
   if not data:break
   p=out/f'gap_source6_final.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)))
 m={k:v for k,v in description.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),tracked_payload_files=len(members),members_manifest_sha256=sha(payload/'MEMBERS.json'),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
def verify(out):
 import hashlib
 m=json.loads((out/'MANIFEST.json').read_text());rows=json.loads((out/'READBACK_INPUT.json').read_text());assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as stream:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert c['name']==r['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'];stream.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];names={r['path'] for r in members}|{'MEMBERS.json'}
  assert len(members)==m['tracked_payload_files'] and set(tar.getnames())==names and len(tar.getmembers())==len(names)
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 r=dict(state='remote_archive_and_all_members_verified',chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(names),archive_sha256=m['archive_sha256'],models_in_this_archive=m['current_archive_models'],full_set_requires_prior_archives=True)
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify-readback']);p.add_argument('--out',type=Path,required=True);args=p.parse_args();(build if args.action=='build' else verify)(args.out)
