"""Remove only already recovered, SHA-verified transport duplicates; no payloads."""
from pathlib import Path
import hashlib,json,shutil,datetime
OUT=Path(__file__).resolve().parent/'TRANSPORT_RECLAIM.json'
assert not OUT.exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
targets=[]
for root in ['/workspace/uwb-resnet18-roundoff-audit-20261102/complete_backup','/workspace/uwb-arsenal-parallel-pilot-20261102/complete_backup']:
 p=Path(root);m=json.loads((p/'MANIFEST.json').read_bytes());r=json.loads((p/'READBACK_VERIFIED.json').read_bytes());rows=json.loads((p/'READBACK_INPUT.json').read_bytes());receipt=json.loads((p/'DRIVE_BACKUP_RECEIPT.json').read_bytes())
 assert r['state']=='remote_archive_and_all_members_verified' and r['complete_modelset_verified'] and r['archive_sha256']==m['archive_sha256']
 assert len(rows)==r['chunks_verified']==len(m['chunks']) and receipt
 for name in [m['archive_name'],'remote_reassembled.tar.gz']:targets.append((p/name,m['archive_bytes'],m['archive_sha256']))
 for c,row in zip(m['chunks'],rows):
  assert all(c[k]==row[k] for k in ['name','bytes','sha256']) and row['download_verified'] and row['raw_fetch'] and row['include_base64']==False
  assert row['metadata']['shared']==False and row['metadata']['permissions']==[{'role':'owner','type':'user'}]
  targets.extend([(p/c['name'],c['bytes'],c['sha256']),(Path(row['path']),c['bytes'],c['sha256'])])
seen=set();verified=[]
for p,n,h in targets:
 assert p.is_file() and not p.is_symlink() and p not in seen and (str(p).startswith('/workspace/attachments/') or '/complete_backup/' in str(p));seen.add(p)
 assert p.stat().st_size==n and sha(p)==h
 verified.append(dict(path=str(p),bytes=n,sha256=h))
before=shutil.disk_usage('/workspace').free
record=dict(state='verified_transport_only_deletion_intent',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=verified,bytes=sum(x['bytes'] for x in verified),free_before=before,original_data_source_models_payloads_receipts_unchanged=True,cloud_objects_unchanged=True)
OUT.write_text(json.dumps(record,indent=2)+'\n')
for x in verified:Path(x['path']).unlink()
record.update(state='complete_verified_transport_duplicates_removed',free_after=shutil.disk_usage('/workspace').free)
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='files'}))
