"""Immutable first-two-fold insurance only; no fitting or diagnostic scoring."""
from pathlib import Path
import hashlib,json,shutil,tarfile,time,sys
B=Path('/workspace/uwb-hinception-sixfold-20261031')
OUT=Path(__file__).resolve().parent/'snapshot01'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
 sys.path.insert(0,str(B));from readiness import ready
 cfg=read(B/'PROTOCOL.json');ready(cfg)
 OUT.mkdir(exist_ok=False);payload=OUT/'payload';payload.mkdir();members=[]
 def copy(p,n):
  p=Path(p);dest=payload/n;dest.parent.mkdir(parents=True,exist_ok=True);h=sha(p);shutil.copyfile(p,dest);assert sha(p)==sha(dest)==h;members.append(dict(path=str(n),bytes=dest.stat().st_size,sha256=h))
 def generated(n,obj):
  p=payload/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n');members.append(dict(path=str(n),bytes=p.stat().st_size,sha256=sha(p)))
 frozen=read(B/'FROZEN_HASHES.json')
 for n in frozen:
  if not Path(n).is_absolute():copy(B/n,Path('study')/n)
 for n in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','LAUNCH_RECEIPT.json']:copy(B/n,Path('study')/n)
 states={}
 for held in [2,3]:
  root=B/'run01'/('held'+str(held));s=read(root/'STATE.json');assert s['state']=='training' and s['total_steps']>=1 and not s['diagnostic_labels_read'];states[str(held)]=s
  # These are immutable once atomically written, unlike active epoch logs/last.
  for n in ['IDENTITY.json','split.json','EVAL_BATCH_IDS.json','initial.pt','first_update.pt']:copy(root/n,Path('run01')/('held'+str(held))/n)
 generated(Path('OBSERVED_LIVE_STATES.json'),states)
 generated(Path('OBSERVED_QUEUE_STATE.json'),read(B/'STATE.json'))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for i,n in enumerate(cfg['dependencies_text']):copy(n,Path('dependencies/text')/(str(i)+'_'+Path(n).name))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 up=Path(cfg['upstream_workspace'])
 for n in ['COMPLETE_AUDIT.json','complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json','complete_backup_20261031/READBACK_VERIFIED.json']:copy(up/n,Path('reused_held1')/n)
 copy('/workspace/uwb-hinception-final-publication-20261031/uwb_direction1/results/cloud_20261031_hinception_pilot/complete_backup/CLOUD_DOWNLOAD_INDEX.json',Path('reused_held1/CLOUD_DOWNLOAD_INDEX.json'))
 copy(__file__,Path('preservation/preserve.py'))
 snapshot=dict(state='immutable_source_confirmation_first_two_update_insurance',scope='held2_held3_initial_first_update_only',observed_unix=time.time(),members=members,raw_arrays_included=False,source_maps_included=False,all_completed_models_included=False,full_job_resume_snapshot=False,full_replay_audits_complete=False,limitations='Only held2/held3 immutable initial+first native/model/Adam/BN/metrics/callback/RNG states, IDs/code and reused held1 private recovery reference; no latest/selected/final models, no whole-job resume or full-fold audit claim. Complete new modelset must be preserved separately after all five audits. No values/labels fitted or scored by preservation.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2)+'\n');prefix='hinception_source6_first2_20261031';archive=OUT/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  for i in range(1000):
   data=f.read(7*1024*1024)
   if not data:break
   p=OUT/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)))
 m={k:v for k,v in snapshot.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(OUT/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(bytes=m['archive_bytes'],chunks=len(chunks),payload_members=len(members))))
def verify():
 m=read(OUT/'MANIFEST.json');rows=read(OUT/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=OUT/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(m['chunks'],rows):
   p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256'];f.write(p.read_bytes())
 assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
 with tarfile.open(archive) as tar:
  data=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];members=json.loads(data)['members'];expected={r['path'] for r in members}|{'MEMBERS.json'};assert len(members)==m['tracked_payload_files'] and len(tar.getmembers())==len(expected) and set(tar.getnames())==expected
  for r in members:
   data=tar.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 result=dict(state='remote_archive_and_all_members_verified',scope=m['scope'],chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),archive_sha256=m['archive_sha256'],complete_modelset_verified=False);(OUT/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':assert sys.argv[1:] in [['build'],['verify']];build() if sys.argv[1]=='build' else verify()
