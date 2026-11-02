"""Exercise chunk/member recovery path using synthetic bytes, no model/data."""
import json,tempfile,tarfile,unittest
from pathlib import Path
import backup
class Roundtrip(unittest.TestCase):
 def test_ordered_raw_chunks_and_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);payload=b/'payload';payload.mkdir();raw=b'synthetic-state-bytes';(payload/'state.bin').write_bytes(raw)
   meta=dict(scope='initial_first_only_not_latest_or_complete',complete_modelset=False,members=[dict(path='state.bin',bytes=len(raw),sha256=backup.sha(payload/'state.bin'))])
   (payload/'MEMBERS.json').write_text(json.dumps(meta));arc=b/'local.tar.gz'
   with tarfile.open(arc,'w:gz') as t:
    for n in ['MEMBERS.json','state.bin']:t.add(payload/n,arcname=n)
   data=arc.read_bytes();chunks=[];rows=[]
   for i,raw in enumerate([data[:len(data)//2],data[len(data)//2:]]):
    p=b/('part'+str(i));p.write_bytes(raw);chunks.append(dict(name=p.name,bytes=len(raw),sha256=backup.sha(p)));rows.append(dict(name=p.name,path=str(p)))
   manifest=dict(scope=meta['scope'],complete_modelset=False,chunks=chunks,archive_bytes=len(data),archive_sha256=backup.sha(arc),members_manifest_sha256=backup.sha(payload/'MEMBERS.json'),tracked_payload_files=1)
   (b/'MANIFEST.json').write_text(json.dumps(manifest));(b/'READBACK_INPUT.json').write_text(json.dumps(rows));backup.verify(b)
   proof=json.loads((b/'READBACK_VERIFIED.json').read_bytes());self.assertEqual(proof['archive_members'],2);self.assertFalse(proof['complete_modelset_verified'])
   Path(rows[0]['path']).write_bytes(b'corrupt')
   with self.assertRaises(AssertionError):backup.verify(b)
if __name__=='__main__':unittest.main()
