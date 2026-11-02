"""Preserve only an immutable finished GAP fold; never decode held predictions.

Live training is read-only. This is not a complete-job resume snapshot.
"""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def build(held,out):
 assert held in range(2,7)
 cfg=json.loads((BASE/'PROTOCOL.json').read_text());run=BASE/'run01';state=json.loads((run/'STATE.json').read_text());assert held in state['completed_folds']
 fold=run/f'held{held}';done=json.loads((fold/'FOLD_COMPLETE.json').read_text());assert done['held']==held and done['epochs']==64
 assert sha(fold/'best.pt')==done['checkpoint_sha256'] and sha(fold/'predictions_unscored.npz')==done['prediction_sha256']
 for index in ('FROZEN_HASHES.json','AUDIT_HASHES.json'):
  for n,h in json.loads((BASE/index).read_text()).items():assert sha(BASE/n)==h,n
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);before=sha(source);shutil.copyfile(source,dest);assert sha(source)==sha(dest)==before,str(source)
  members.append(dict(path=str(name),bytes=dest.stat().st_size,sha256=before))
 for p in sorted(BASE.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.md','.log'):copy(p,Path('study')/p.name)
 for name in ('IDENTITY.json','REUSED_HELD1.json'):copy(run/name,Path('run_metadata')/name)
 for p in sorted(fold.iterdir()):
  assert p.is_file() and p.suffix!='.tmp';copy(p,Path(f'held{held}')/p.name)
 upstream=Path(cfg['reused_held1']['directory']).parents[1]
 copy(upstream/'adapter.py',Path('dependencies/timemil/adapter.py'))
 for directory in ('vendor/TimeMIL/models','vendor/site-packages'):
  for p in sorted((upstream/directory).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p,Path('dependencies/timemil')/p.relative_to(upstream))
 for p in sorted((upstream/'vendor/TimeMIL').glob('*')):
  if p.is_file() and (p.name.upper().startswith(('LICENSE','LICENCE','README'))):copy(p,Path('dependencies/timemil')/p.relative_to(upstream))
 copy(Path(cfg['source_manifest']),Path('dependencies/source_ids.npz'));assert sha(payload/'dependencies/source_ids.npz')==cfg['source_manifest_sha256']
 for key in ('upstream_audit','localcnn_audit','upstream_private_backup'):copy(Path(cfg[key]),Path('dependencies')/(key+'.json'))
 for h,ref in cfg['cnn_references'].items():
  assert sha(ref['path'])==ref['sha256'];copy(Path(ref['path']),Path('dependencies/cnn_predictions')/f'held{h}.npz')
 copy(Path('/workspace/uwb-multirocket-20261029/launch.py'),Path('dependencies/independent_supervisor.py'))
 snapshot=dict(state='immutable_completed_fold_snapshot',held=held,epochs=64,observed_unix=time.time(),live_state_context=state,members=members,
               diagnostic_values_decoded=False,real_sample_or_label_values_read=False,real_fitting_or_forward=False,
               raw_arrays_included=False,source_maps_included=False,live_incomplete_fold_included=False,full_job_resume_snapshot=False,
               current_archive_models=[held],held1_models_require_prior_backup=True,
               limitations='Only immutable completed fold and code/IDs/reference prerequisites. Upstream complete private receipt recovers reused held1 tensors; original input migration index restores raw data and exact saved RMS/reshape regenerates maps. Current live fold and remaining folds absent; never restart/duplicate the alive worker from this bundle.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));archive=out/f'gap_source6_held{held}.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as stream:
  for i in range(1000):
   data=stream.read(7*1024*1024)
   if not data:break
   p=out/f'gap_source6_held{held}.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)))
 manifest={k:v for k,v in snapshot.items() if k not in ('members','live_state_context')}
 manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),tracked_payload_files=len(members),members_manifest_sha256=sha(payload/'MEMBERS.json'),chunks=chunks)
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
def verify_readback(out):
 m=json.loads((out/'MANIFEST.json').read_text());rows=json.loads((out/'READBACK_INPUT.json').read_text());assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as stream:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert c['name']==r['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'];stream.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];expected={r['path'] for r in members}|{'MEMBERS.json'}
  assert len(members)==m['tracked_payload_files'] and set(tar.getnames())==expected and len(tar.getmembers())==len(expected)
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 result=dict(state='remote_archive_and_all_members_verified',held=m['held'],epochs=m['epochs'],chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'])
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify-readback']);parser.add_argument('--held',type=int);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 if args.action=='build':build(args.held,args.out)
 else:verify_readback(args.out)
