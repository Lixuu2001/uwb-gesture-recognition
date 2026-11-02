import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import full_view_heads as h


class FullViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_crossuser_masks_do_not_make_same_user_same_class_negative(self):
        labels = torch.tensor([0,0,0,1])
        users = torch.tensor([1,1,2,1])
        pos, eligible = h.contrast_masks(labels, users, True)
        self.assertTrue(pos[0,2])
        self.assertFalse(eligible[0,1])
        self.assertTrue(eligible[0,3])
        self.assertFalse(pos.diag().any())

    def test_ordinary_loss_matches_author_reference(self):
        source = Path(__file__).resolve().parents[1]/'research/sources/supcontrast_losses.py.txt'
        ns = {}
        exec(compile(source.read_text(), str(source), 'exec'), ns)
        torch.manual_seed(2026)
        z = torch.randn(12, 16, requires_grad=True)
        labels = torch.arange(6).repeat_interleave(2)
        users = torch.arange(12)
        ours = h.supcon(z, labels, users, temperature=.1)
        reference = ns['SupConLoss'](temperature=.1, base_temperature=.1)(F.normalize(z,dim=1)[:,None],labels)
        torch.testing.assert_close(ours, reference, atol=1e-6, rtol=1e-6)
        ours.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.norm().item(), 0)

    def test_no_positives_has_finite_zero_gradient(self):
        z = torch.randn(4,16,requires_grad=True)
        loss = h.supcon(z, torch.zeros(4,dtype=torch.long),torch.ones(4,dtype=torch.long),True)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_hidden_view_invariance(self):
        model = h.Head().eval()
        x = torch.randn(4,3,256)
        changed = x.clone(); changed[:,:2] += 1000
        torch.testing.assert_close(model(x,h.MASKS[2])[0],model(changed,h.MASKS[2])[0],rtol=0,atol=0)

    def test_epoch_sampling_deterministic_and_cell_balanced(self):
        y = np.tile(np.arange(12), 24)
        users = np.repeat(np.arange(1,7), 48)
        a = h.sample_epoch(y, users, 2026, 1, 1)
        np.testing.assert_array_equal(a,h.sample_epoch(y,users,2026,1,1))
        for batch in a:
            self.assertEqual(len(np.unique(batch)),len(batch))
            for u in range(1,7):
                for c in range(12):
                    self.assertEqual(np.sum((y[batch]==c)&(users[batch]==u)),2)

    def test_interrupt_resume_matches_uninterrupted_training(self):
        rng = np.random.default_rng(9)
        y = np.tile(np.arange(12),14)
        users = np.repeat(np.arange(1,8),24)
        roles = np.repeat([0,1,2],[144,12,12])
        x = rng.normal(size=(len(y),3,256)).astype('float32')
        ids = np.arange(len(y))
        config = {'model_seed':2026,'optimizer':{'lr':.001,'weight_decay':.0001},
                  'epochs':2,'min_epochs':2,'patience':8,'supcon':{'temperature':.1,'weight':.05}}
        identity = {'fixture':'resume-test'}
        with tempfile.TemporaryDirectory() as folder:
            clean, resumed = Path(folder)/'clean', Path(folder)/'resumed'
            original_save = h.atomic_save
            def interrupt_after_save(path, obj):
                original_save(path,obj)
                if path.name == 'progress.pt':
                    raise InterruptedError('fixture interrupt after durable epoch checkpoint')
            with patch.object(h,'atomic_save',side_effect=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    h.fit_arm(resumed,config,identity,'full_crossuser',1,x,y,users,roles,ids)
            h.fit_arm(resumed,config,identity,'full_crossuser',1,x,y,users,roles,ids)
            h.fit_arm(clean,config,identity,'full_crossuser',1,x,y,users,roles,ids)
            with np.load(clean/'predictions.npz') as a,np.load(resumed/'predictions.npz') as b:
                np.testing.assert_array_equal(a['logits'],b['logits'])
            a=torch.load(clean/'progress.pt',weights_only=False)
            b=torch.load(resumed/'progress.pt',weights_only=False)
            for k in a['model']:
                torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
