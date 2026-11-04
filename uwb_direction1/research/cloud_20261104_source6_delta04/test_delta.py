import tempfile,unittest,json
from pathlib import Path
from delta import read,write,sha,create,apply
class DeltaIntegrity(unittest.TestCase):
    def fixture(self,d):
        b=Path(d)/'base';c=Path(d)/'current';b.mkdir();c.mkdir()
        for name,value in [('same',b'unchanged'),('change',b'old'),('prior_forensics',b'keep original')]: (b/name).write_bytes(value)
        for name,value in [('same',b'unchanged'),('change',b'new'),('new_file',b'new payload')]: (c/name).write_bytes(value)
        for p in [b,c]:write(p/'PAYLOAD_HASHES.json',{f.name:sha(f) for f in p.iterdir()})
        delta=Path(d)/'delta';create(b,c,delta,'synthetic_base_sha');return b,c,delta
    def test_exact_roundtrip_and_original_forensics_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            b,c,x=self.fixture(d);before={p.name:p.read_bytes() for p in b.iterdir()};out=Path(d)/'restored';apply(b,x,out)
            for n,h in read(c/'PAYLOAD_HASHES.json').items():self.assertEqual(sha(out/n),h)
            self.assertFalse((out/'prior_forensics').exists());self.assertEqual(before,{p.name:p.read_bytes() for p in b.iterdir()})
    def test_corrupt_base_or_changed_payload_refused(self):
        with tempfile.TemporaryDirectory() as d:
            b,c,x=self.fixture(d);(x/'changed/change').write_bytes(b'bad')
            with self.assertRaises(AssertionError):apply(b,x,Path(d)/'bad_out')
            self.assertFalse((Path(d)/'bad_out').exists())
        with tempfile.TemporaryDirectory() as d:
            b,c,x=self.fixture(d);(b/'same').write_bytes(b'bad')
            with self.assertRaises(AssertionError):apply(b,x,Path(d)/'bad_out')
    def test_path_traversal_refused_before_any_output(self):
        with tempfile.TemporaryDirectory() as d:
            b,c,x=self.fixture(d);write(x/'CURRENT_PAYLOAD_HASHES.json',{'../escape':'bad'});p=read(x/'DELTA.json');p['current_payload_table_sha256']=sha(x/'CURRENT_PAYLOAD_HASHES.json');write(x/'DELTA.json',p)
            with self.assertRaises(AssertionError):apply(b,x,Path(d)/'bad_out')
            self.assertFalse((Path(d)/'bad_out').exists())
if __name__=='__main__':unittest.main(verbosity=2)
