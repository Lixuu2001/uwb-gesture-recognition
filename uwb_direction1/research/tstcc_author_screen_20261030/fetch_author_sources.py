"""Restore pinned public author bytes and validate both digests; no real data."""
import hashlib,json,urllib.request
from pathlib import Path
B=Path(__file__).resolve().parent
if __name__=='__main__':
 index=json.loads((B/'SOURCE_INDEX.json').read_text());commit=index['commit']
 for row in index['files']:
  p=B/'vendor/TS-TCC'/row['path']
  if p.exists():data=p.read_bytes()
  else:
   url='https://raw.githubusercontent.com/'+index['repository']+'/'+commit+'/'+row['path']
   with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
  assert hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
  assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1'],row['path']
  if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 print('Pinned10authorfiles verified; no data/model/fitting invoked')
