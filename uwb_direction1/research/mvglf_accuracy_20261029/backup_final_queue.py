"""Archive completed CNN/MultiRocket and MiniRocket controls, excluding large maps."""
from pathlib import Path
import hashlib,io,json,tarfile,datetime
B=Path('/workspace/uwb-final-queue-backup-20261029')
def sha(v):return hashlib.sha256(v).hexdigest()
def main():
 B.mkdir(exist_ok=False);members={}
 roots=[('mvglf',Path('/workspace/uwb-mvglf-20261029')),('multirocket',Path('/workspace/uwb-multirocket-20261029'))]
 for prefix,root in roots:
  for p in root.iterdir():
   if p.is_file() and p.suffix in {'.py','.json','.md','.txt','.log'}:members[prefix+'/'+p.name]=p.read_bytes()
  for name in ['run01','continuation01','continuation02','supervisor01']:
   d=root/name
   if not d.exists():continue
   for p in d.rglob('*'):
    if p.is_file() and p.suffix not in {'.npy','.tmp'} and p.name not in {'worker.lock','controller.lock'}:
     members[prefix+'/'+str(p.relative_to(root))]=p.read_bytes()
 # Existing six-fold MiniRocket controls, including exact source manifest/IDs.
 control=Path('/workspace/uwb-accuracy-loop-20261029/run01')
 for p in control.rglob('*'):
  if p.is_file() and p.suffix not in {'.npy','.tmp'}:members['minirocket_control/run01/'+str(p.relative_to(control))]=p.read_bytes()
 code=Path('/workspace/uwb-accuracy-loop-20261029/vendor/minirocket/code/minirocket_multivariate.py')
 members['minirocket_control/minirocket_multivariate.py']=code.read_bytes()
 for prefix,root in roots:
  state=json.loads((root/'run01/STATE.json').read_text());assert state['state']=='complete'
  audit=root/('COMPLETE_AUDIT.json' if prefix=='mvglf' else 'COMPLETE_RESULT_AUDIT.json')
  assert json.loads(audit.read_text())['state']=='pass'
 manifest=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='completed CNN and MultiRocket fixed source pilots plus existing MiniRocket six-source controls',raw_radar_arrays_included=False,large_derived_maps_or_features_included=False,private_ids_models_and_predictions_included=True,
  data_dependency='/workspace/uwb-restored-20261020; original persistent private Drive package remains separate',
  members={n:dict(bytes=len(v),sha256=sha(v)) for n,v in members.items()})
 archive=B/'final_queue.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for n,v in members.items():
   info=tarfile.TarInfo(n);info.size=len(v);info.mode=0o600;tar.addfile(info,io.BytesIO(v))
  v=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('MEMBERS.json');info.size=len(v);tar.addfile(info,io.BytesIO(v))
 raw=archive.read_bytes();chunks=[]
 for i,start in enumerate(range(0,len(raw),7*1024*1024)):
  v=raw[start:start+7*1024*1024];p=B/f'final_queue.part{i:03d}';p.write_bytes(v);chunks.append(dict(name=p.name,bytes=len(v),sha256=sha(v)))
 with tarfile.open(archive,'r:gz') as tar:
  for n,record in manifest['members'].items():assert sha(tar.extractfile(n).read())==record['sha256']
 result=dict(manifest=manifest,archive_bytes=len(raw),archive_sha256=sha(raw),chunks=chunks,local_members_verified=True)
 (B/'MANIFEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(members=len(members),bytes=len(raw),chunks=len(chunks),sha256=result['archive_sha256'])))
if __name__=='__main__':main()
