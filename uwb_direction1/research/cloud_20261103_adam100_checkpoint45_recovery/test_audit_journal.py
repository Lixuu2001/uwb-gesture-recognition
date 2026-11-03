import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit_journal as journal
from adam_engine import torch,record_epoch
from trial_common import read,write

def state(epoch):
    return dict(model={'counter':torch.tensor(epoch)},optimizer={'step':epoch},
        order_rng=torch.tensor([epoch],dtype=torch.uint8),
        torch_rng=torch.tensor([epoch+1],dtype=torch.uint8),epoch=epoch,updates=235*epoch)
def row(s):
    return record_epoch(dict(epoch=s['epoch'],updates=s['updates'],validation_macro=0.,
        selection='fixed_final_epoch100_not_best_validation',diagnostic_labels_read=False,
        users7_8_labels_used=False),s)
class Journal(unittest.TestCase):
    def test_two_committed_epochs_restore_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'producer';out.mkdir();audit=root/'audit'
            for e in [1,2]:
                s=state(e);write(out/f'epoch{e:03d}.json',row(s));journal.commit(audit,out,row(s),s)
            self.assertEqual(journal.semantic(journal.load(audit,out)),journal.semantic(state(2)))
    def test_interruption_before_progress_preserves_previous_slot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'producer';out.mkdir();audit=root/'audit'
            for e in [1,2]:write(out/f'epoch{e:03d}.json',row(state(e)))
            journal.commit(audit,out,row(state(1)),state(1));original=journal.write
            def fail(path,value):
                if Path(path).name=='PROGRESS.json':raise OSError('synthetic crash before commit')
                original(path,value)
            with patch.object(journal,'write',side_effect=fail):
                with self.assertRaises(OSError):journal.commit(audit,out,row(state(2)),state(2))
            self.assertEqual(journal.load(audit,out)['epoch'],1)
    def test_changed_journal_or_producer_record_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'producer';out.mkdir();audit=root/'audit'
            s=state(1);write(out/'epoch001.json',row(s));journal.commit(audit,out,row(s),s)
            r=read(out/'epoch001.json');r['model_sha256']='0'*64;write(out/'epoch001.json',r)
            with self.assertRaises(AssertionError):journal.load(audit,out)
if __name__=='__main__':unittest.main(verbosity=2)
