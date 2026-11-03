import tempfile,unittest,json,hashlib
from pathlib import Path
import engine as e
e.setup('shuffle')
from journal import StepLog,commit,load
from common import read,write
class JournalChecks(unittest.TestCase):
    def objects(self):
        m=e.torch.nn.Linear(3,2);o=e.torch.optim.Adam(m.parameters());g=e.torch.Generator().manual_seed(1)
        return m,o,g
    def test_slot_survives_orphan_and_exact_suffix_preserves_raw_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Path(folder);m,o,g=self.objects();log=StepLog(b/'steps.jsonl')
            s=e.capture(m,o,g,0,0);commit(b,s,e.record(s,0,0),log.prefix())
            row=dict(update=1,loss=.5);log.append(row)
            s=e.capture(m,o,g,1,1);commit(b,s,e.record(s,1,1),log.prefix())
            log.append(dict(update=2,loss=.25));log.close();original=(b/'steps.jsonl').read_bytes()
            # A crash after writing the next slot must leave the prior commit valid.
            e.atomic_save(b/'slot0.pt',e.capture(m,o,g,2,2))
            recovered,p=load(b);self.assertEqual(recovered['epoch'],1)
            log=StepLog(b/'steps.jsonl',1,p['prefix_bytes'],p['prefix_sha256'])
            log.append(dict(update=2,loss=.25));self.assertEqual((b/'steps.jsonl').read_bytes(),original)
            with self.assertRaises(AssertionError):log.append(dict(update=9,loss=.1))
            log.close()
    def test_changed_uncommitted_suffix_or_committed_prefix_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'steps';log=StepLog(p);log.append(dict(update=1,loss=.5));prefix=log.prefix();log.append(dict(update=2,loss=.3));log.close()
            log=StepLog(p,1,prefix['prefix_bytes'],prefix['prefix_sha256'])
            with self.assertRaises(AssertionError):log.append(dict(update=2,loss=.4))
            log.close();p.write_bytes(p.read_bytes().replace(b'0.5',b'0.6'))
            with self.assertRaises(AssertionError):StepLog(p,1,prefix['prefix_bytes'],prefix['prefix_sha256'])
    def test_checkpoint_and_chain_tamper_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Path(folder);m,o,g=self.objects();log=StepLog(b/'steps')
            s=e.capture(m,o,g,0,0);commit(b,s,e.record(s,0,0),log.prefix());log.close()
            path=b/'slot0.pt';path.write_bytes(path.read_bytes()+b'bad')
            with self.assertRaises(AssertionError):load(b)
if __name__=='__main__':unittest.main(verbosity=2)
