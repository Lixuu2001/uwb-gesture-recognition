import unittest,tempfile,hashlib,json
from pathlib import Path
from stream_archive import build,verify,replace_verified_download
class Stream(unittest.TestCase):
 def setup_fixture(self,d):
  p=Path(d);a=p/'model';b=p/'ids';a.write_bytes(bytes(range(256))*31);b.write_bytes(b'private-synthetic-ids');m=build(p/'archive','fixture',{'model/full.pt':a,'private/ids.npz':b},'synthetic',chunk_size=71,limit=100000);rows=[dict(c,path=str(p/'archive'/c['name'])) for c in m['chunks']];return m,rows
 def test_multichunk_all_members_and_atomic_download_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   m,rows=self.setup_fixture(d);self.assertGreater(len(rows),3)
   for r in rows:
    download=Path(d)/'download';download.write_bytes(Path(r['path']).read_bytes());replace_verified_download(r['path'],download,r['bytes'],r['sha256']);self.assertFalse(download.exists())
   self.assertEqual(verify(m,rows)['archive_members'],3)
 def test_corruption_and_missing_or_reordered_chunks_block(self):
  with tempfile.TemporaryDirectory() as d:
   m,rows=self.setup_fixture(d)
   with self.assertRaises(AssertionError):verify(m,rows[:-1])
   with self.assertRaises(AssertionError):verify(m,list(reversed(rows)))
   p=Path(rows[-1]['path']);b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
   with self.assertRaises((AssertionError,OSError,EOFError)):verify(m,rows)
 def test_incomplete_model_manifest_and_path_traversal_block(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a=p/'file';a.write_bytes(b'x')
   with self.assertRaises(AssertionError):build(p/'bad','fixture',{'../escape':a},'synthetic')
if __name__=='__main__':unittest.main(verbosity=2)
