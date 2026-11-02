"""Synthetic byte-fixture tests only: no real arrays, labels, fits or model loads."""
from pathlib import Path
from unittest.mock import patch
import contextlib,io,json,tarfile,tempfile,unittest
import complete_modelset as m
class Tests(unittest.TestCase):
 def fixture(self,root):
  out=root/'archive';(out/'payload').mkdir(parents=True);p=out/'payload/fixture.txt';p.write_text('synthetic bytes, not a trained model\n');members=[dict(path='fixture.txt',bytes=p.stat().st_size,sha256=m.sha(p))];manifest=m.pack(out,'complete_six_models_and_five_new_full_replay_audits',members,'synthetic_fixture');data=(out/manifest['archive_name']).read_bytes();parts=[data[:len(data)//2],data[len(data)//2:]];manifest['chunks']=[]
  for i,data in enumerate(parts):
   p=out/f'synthetic_fixture.part{i:03d}';p.write_bytes(data);manifest['chunks'].append(dict(name=p.name,bytes=len(data),sha256=m.sha(p)))
  (out/'MANIFEST.json').write_text(json.dumps(manifest));rows=[dict(name=c['name'],path=str(out/c['name'])) for c in manifest['chunks']];(out/'READBACK_INPUT.json').write_text(json.dumps(rows));return out,manifest
 def test_active_work_refuses_before_hash_or_archive_creation(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d)/'study';b.mkdir();(b/'STATE.json').write_text(json.dumps(dict(state='running_unscored_source_confirmation')));out=Path(d)/'must_not_exist'
   with patch.object(m,'sha',side_effect=AssertionError('No artifact hashes before completion')) as hasher:
    with self.assertRaisesRegex(RuntimeError,'active/incomplete'):m.build(b,out)
    hasher.assert_not_called();self.assertFalse(out.exists())
 def test_valid_synthetic_readback_checks_all_members(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d))
   with contextlib.redirect_stdout(io.StringIO()):proof=m.verify_readback(out)
   self.assertEqual(proof['tracked_payload_members_verified'],1);self.assertEqual(proof['archive_members'],2);self.assertEqual(proof['archive_sha256'],manifest['archive_sha256'])
 def test_changed_chunk_refuses_without_success_proof(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));p=out/manifest['chunks'][0]['name'];data=bytearray(p.read_bytes());data[len(data)//2]^=1;p.write_bytes(data)
   with self.assertRaisesRegex(RuntimeError,'Changed artifact'):m.verify_readback(out)
   self.assertFalse((out/'READBACK_VERIFIED.json').exists())
 def test_changed_payload_fails_even_with_matching_external_archive(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));archive=out/'changed.tar.gz';(out/'payload/fixture.txt').write_text('corruption of synthetic member')
   with tarfile.open(archive,'w:gz') as tar:
    for p in sorted((out/'payload').iterdir()):tar.add(p,arcname=p.name)
   with self.assertRaisesRegex(RuntimeError,'Corrupt payload member'):m.verify_members(archive,manifest['members_manifest_sha256'],1)
 def test_duplicate_tar_member_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));archive=out/'duplicate.tar.gz'
   with tarfile.open(archive,'w:gz') as tar:
    for p in sorted((out/'payload').iterdir()):tar.add(p,arcname=p.name)
    tar.add(out/'payload/fixture.txt',arcname='fixture.txt')
   with self.assertRaisesRegex(RuntimeError,'Duplicate/non-file'):m.verify_members(archive,manifest['members_manifest_sha256'],1)
 def test_partial_manifest_never_becomes_complete_proof(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));manifest['scope']='first_update_only';(out/'MANIFEST.json').write_text(json.dumps(manifest))
   with self.assertRaisesRegex(RuntimeError,'Partial manifest'):m.verify_readback(out)
   self.assertFalse((out/'READBACK_VERIFIED.json').exists())
 def test_unsafe_member_name_refuses(self):
  for name in ['../escape','/absolute','a/../escape']:
   with self.assertRaisesRegex(RuntimeError,'Unsafe member name'):m.relative_name(name)
 def test_swapped_chunk_order_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));rows=m.read(out/'READBACK_INPUT.json');(out/'READBACK_INPUT.json').write_text(json.dumps(rows[::-1]))
   with self.assertRaisesRegex(RuntimeError,'Wrong readback name/size'):m.verify_readback(out)
   self.assertFalse((out/'READBACK_VERIFIED.json').exists())
 def test_partial_embedded_scope_refuses_even_when_transport_hashes_match(self):
  with tempfile.TemporaryDirectory() as d:
   out,manifest=self.fixture(Path(d));p=out/'payload/MEMBERS.json';embedded=m.read(p);embedded['scope']='first_update_only';p.write_text(json.dumps(embedded));archive=out/'changed_embedded_scope.tar.gz'
   with tarfile.open(archive,'w:gz') as tar:
    for p in sorted((out/'payload').iterdir()):tar.add(p,arcname=p.name)
   manifest.update(archive_bytes=archive.stat().st_size,archive_sha256=m.sha(archive),members_manifest_sha256=m.sha(out/'payload/MEMBERS.json'),chunks=[dict(name=archive.name,bytes=archive.stat().st_size,sha256=m.sha(archive))]);(out/'MANIFEST.json').write_text(json.dumps(manifest));(out/'READBACK_INPUT.json').write_text(json.dumps([dict(name=archive.name,path=str(archive))]))
   with self.assertRaisesRegex(RuntimeError,'Changed archive/external scope'):m.verify_readback(out)
   self.assertFalse((out/'READBACK_VERIFIED.json').exists())
if __name__=='__main__':unittest.main()
