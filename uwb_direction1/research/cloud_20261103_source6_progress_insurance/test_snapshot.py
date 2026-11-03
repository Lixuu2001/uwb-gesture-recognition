import unittest,tempfile,hashlib,json
from pathlib import Path
from unittest.mock import patch
import snapshot as s
class SnapshotGuards(unittest.TestCase):
    def fixture(self,root,tail=b''):
        root.mkdir();(root/'slot0.pt').write_bytes(b'synthetic state')
        row=(json.dumps(dict(update=1,epoch=0))+'\n').encode();(root/'steps.jsonl').write_bytes(row+tail)
        record=dict(epoch=0,updates=1,state_sha256='synthetic_semantic_hash');s.write(root/'epoch000.json',record)
        journal=dict(epoch=0,previous_journal_sha256=None,record_sha256=s.sha(root/'epoch000.json'),state_sha256=record['state_sha256'],checkpoint_sha256=s.sha(root/'slot0.pt'));s.write(root/'journal000.json',journal)
        p=dict(epoch=0,updates=1,slot='slot0.pt',checkpoint_sha256=s.sha(root/'slot0.pt'),journal_sha256=s.sha(root/'journal000.json'),prefix_bytes=len(row),prefix_sha256=hashlib.sha256(row).hexdigest());s.write(root/'PROGRESS.json',p);return p
    def test_committed_cut_retains_complete_suffix_and_raw_partial(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';tail=b'{"update": 2}\n{"update":';p=self.fixture(a,tail);out=Path(d)/'snapshot';r=s.capture_journal(a,out)
            self.assertEqual(r['step_log']['complete_rows'],2);self.assertEqual(r['step_log']['partial_tail_bytes'],10)
            self.assertEqual((a/'steps.jsonl').read_bytes(),(out/'steps_raw_cut.bin').read_bytes())
            self.assertTrue((out/'steps.jsonl').read_bytes().endswith(b'\n'));self.assertEqual(s.read(out/'PROGRESS.json'),p)
    def test_corrupt_committed_prefix_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';self.fixture(a);(a/'steps.jsonl').write_bytes(b'{"update": 9,"epoch": 0}\n')
            with self.assertRaises(AssertionError):s.capture_journal(a,Path(d)/'snapshot')
    def test_slot_replacement_with_wrong_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';self.fixture(a);(a/'slot0.pt').write_bytes(b'newer different state')
            with self.assertRaises(AssertionError):s.capture_journal(a,Path(d)/'snapshot')
    def test_progress_race_rejected_without_modifying_source_logs(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';p=self.fixture(a);original=s.copy;raw=(a/'steps.jsonl').read_bytes()
            def racing(src,dest):
                original(src,dest)
                if Path(src).name=='journal000.json':s.write(a/'PROGRESS.json',dict(**p,race_marker=True))
            with patch.object(s,'copy',racing):
                with self.assertRaises(AssertionError):s.capture_journal(a,Path(d)/'snapshot')
            self.assertEqual((a/'steps.jsonl').read_bytes(),raw)
if __name__=='__main__':unittest.main(verbosity=2)
