import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import circular_signal as cs


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_phasor_branch_continuity_and_energy(self):
        eps=1e-6
        d=np.exp(1j*np.array([[np.pi-eps,-np.pi+eps]]))
        x=cs.increment_channels(d)
        self.assertGreater(abs(x[1,0,0]-x[1,0,1]),6)
        self.assertLess(np.linalg.norm(x[3:,0,0]-x[3:,0,1]),3e-6)
        np.testing.assert_allclose(x[3]**2+x[4]**2,x[0]**2,atol=1e-12)
        self.assertTrue(np.array_equal(cs.increment_channels(np.zeros((2,3),complex)),np.zeros((5,2,3))))

    def test_analytic_matches_scipy_and_clip_independence(self):
        from scipy.signal import hilbert
        rng=np.random.default_rng(9)
        for n in (188,189):
            x=rng.normal(size=(2,3,n))
            np.testing.assert_allclose(cs.analytic(x),hilbert(x,axis=-1),atol=1e-12)
        raw=rng.normal(size=(2,3,90,189)).astype('float32')
        together=cs.encode(raw)
        np.testing.assert_array_equal(together[0],cs.encode(raw[:1])[0])
        np.testing.assert_array_equal(cs.encode(np.zeros_like(raw)),np.zeros((2,3,6,90,63)))

    def test_original_angle_equivalence(self):
        # Independent formula from the preserved original preprocessing.
        from scipy.signal import hilbert
        raw=np.random.default_rng(1).normal(size=(2,3,90,189)).astype('float32')
        z=hilbert(raw.astype('float64'),axis=-1)
        z-=z.mean(axis=-2,keepdims=True)
        d=z[...,1:,:]*z[...,:-1,:].conj()
        v=(np.angle(d)*np.sqrt(np.abs(d))).astype('float32')
        v=np.concatenate((v[...,:1,:],v),axis=-2)
        v=(v-v.mean(axis=(-2,-1),keepdims=True))/(v.std(axis=(-2,-1),keepdims=True)+1e-6)
        v=v.astype('float16').astype('float32').reshape(2,3,90,63,3).mean(-1).astype('float16')
        np.testing.assert_array_equal(cs.encode(raw)[:,:,2],v)

    def test_source_order_and_channel_ablation(self):
        labels=np.tile(np.arange(12),8)
        users=np.repeat(np.arange(1,9),12)
        parts=cs.expected_parts(labels,users,3)
        ids=cs.source_order(parts['train'],2026,3,7)
        np.testing.assert_array_equal(np.sort(ids),parts['train'])
        self.assertFalse(np.isin(users[ids],[3,4]).any())
        x=np.ones((2,3,6,90,63),dtype='float16')
        legacy=cs.get_batch(x,[0,1],'legacy')
        self.assertEqual(legacy[:,:,2:].count_nonzero(),0)
        self.assertEqual(cs.get_batch(x,[0,1],'circular').shape,(2,3,4,90,63))

    def test_interrupted_resume_exact(self):
        data=np.random.default_rng(7).normal(size=(10,3,6,90,63)).astype('float16')
        labels=np.arange(10)%3
        users=np.arange(10)%8+1
        parts={'train':np.arange(6),'val':np.arange(6,8),'test':np.arange(8,10)}
        config={'seed':2026,'lr':.001,'weight_decay':.001,'epochs':2,'min_lr':.00001,
                'min_epochs':2,'patience':8,'batch':4}
        identity={'test':'fixed'}
        real_save=cs.atomic_save
        def fail_after_save(path,obj):
            real_save(path,obj)
            if path.name=='progress.pt':raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as folder,patch.object(cs,'expected_parts',return_value=parts):
            full=Path(folder)/'full';resumed=Path(folder)/'resumed'
            cs.fit(full,config,identity,data,labels,users,1,'circular')
            with patch.object(cs,'atomic_save',side_effect=fail_after_save):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    cs.fit(resumed,config,identity,data,labels,users,1,'circular')
            cs.fit(resumed,config,identity,data,labels,users,1,'circular')
            a=torch.load(full/'best.pt',weights_only=False)
            b=torch.load(resumed/'best.pt',weights_only=False)
            self.assertEqual(a['epoch'],b['epoch'])
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]))
            with np.load(full/'predictions.npz') as a,np.load(resumed/'predictions.npz') as b:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])


if __name__=='__main__':unittest.main()
