"""Private complete model, SVM, IDs and learned-artifact preservation; no raw arrays."""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from audit_complete import require_complete,read,sha
B=Path(__file__).resolve().parent
def build(out):
 require_complete(B);audit=read(B/'COMPLETE_AUDIT.json');assert audit['state']=='pass'
 completion=read(B/'AUDIT_COMPLETION_RECEIPT.json');assert completion['state']=='pass' and completion['supervisor_folder']=='audit_supervisor02'
 term=read(B/completion['supervisor_folder']/'TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout'] and completion['termination']==term
 for n,h in read(B/'REPAIRED_AUDIT_HASHES.json').items():assert sha(B/n)==h,n
 for p,h in audit['artifact_hashes'].items():assert sha(p)==h,p
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[]
 def copy(source,name):
  source=Path(source);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);h=sha(source);shutil.copyfile(source,target);assert sha(source)==sha(target)==h;members.append(dict(path=str(name),bytes=target.stat().st_size,sha256=h))
 for p in sorted(B.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.md','.log') and p.name not in ('LAUNCH.log','AUDIT_LAUNCH.log'):copy(p,Path('study')/p.name)
 for p in sorted((B/'run01').rglob('*')):
  if p.is_file() and p.name!='worker.lock':copy(p,Path('run01')/p.relative_to(B/'run01'))
 for folder in ('supervisor01','audit_supervisor01','audit_supervisor02'):
  for p in sorted((B/folder).iterdir()):
   if p.is_file():copy(p,Path(folder)/p.name)
 cfg=read(B/'PROTOCOL.json');screen=Path(cfg['prototype'])
 for name in ('author_adapter.py','readout_adapter.py','SOURCE_INDEX.json','SYNTHETIC_CHECK.json','READOUT_SYNTHETIC_CHECK.json','COST_CHECK.json','COST_POLICY.json'):copy(screen/name,Path('dependencies/prototype')/name)
 for p in sorted((screen/'vendor/ts2vec').rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:copy(p,Path('dependencies/author')/p.relative_to(screen/'vendor/ts2vec'))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,p in enumerate(cfg['prior_evidence_sha256']):copy(p,Path('dependencies/prior_evidence')/(str(i)+'_'+Path(p).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 snapshot=dict(state='completed_TS2Vec600_full_model_and_evidence_snapshot',observed_unix=time.time(),members=members,raw_arrays_included=False,source_maps_included=False,learned_features_and_predictions_included=True,initial_first_last_raw_average_optimizer_RNG_included=True,full_600_update_ID_log_included=True,SVM_model_and_training_CV_IDs_included=True,checkpoint_interval_policy='last.pt atomically overwritten every50updates; initial and first_update separate. Intermediate50step versions were rolling, not retained as separate historical files.',full_job_resume_snapshot=False,limitations='Completed-job model recovery and reproducibility artifacts. No automatic job resume. Original migration restores raw maps, exact RMS regenerates source maps. No new fitting/scoring by backup; source pilot exploratory.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2));archive=out/'ts2vec_complete_20261030.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  for i in range(1000):
   data=f.read(7*1024*1024)
   if not data:break
   p=out/f'ts2vec_complete_20261030.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)))
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
 result=dict(state='remote_archive_and_all_members_verified',chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'])
 (out/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify-readback']);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();(build if args.action=='build' else verify)(args.out)
