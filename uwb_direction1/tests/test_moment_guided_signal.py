import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import moment_guided_signal as m
from probe_range_moments import moments,features


class MomentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(9)
        cls.seq=rng.uniform(5,70,size=(24,3,2,90))
        cls.cache=(rng.normal(size=(24,3,6,90,63)).astype(np.float16),cls.seq)
        cls.y=np.tile(np.arange(3),8);cls.u=np.repeat(np.arange(1,9),3)
        cls.ids=m.expected_parts(cls.y,cls.u,1)['train']

    def test_known_moments_and_invariances(self):
        x=np.zeros((1,3,90,189));x[...,20]=1;x[...,40]=1
        z=moments(x)
        np.testing.assert_allclose(z[:,:,0],30);np.testing.assert_allclose(z[:,:,1],10)
        np.testing.assert_allclose(moments(x*9),z)
        shifted=moments(np.roll(x,7,-1))
        np.testing.assert_allclose(shifted[:,:,0],37);np.testing.assert_allclose(shifted[:,:,1],10)
        self.assertTrue(np.isfinite(moments(x*0)).all())

    def test_source_normalization_ignores_outer_targets(self):
        alt=self.seq.copy();alt[~np.isin(np.arange(24),self.ids)]=1e10
        a=m.MomentNet(self.seq,self.ids,'ordered');b=m.MomentNet(alt,self.ids,'ordered')
        self.assertTrue(torch.equal(a.moment_mean,b.moment_mean))
        self.assertTrue(torch.equal(a.moment_scale,b.moment_scale))

    def test_controls_parameter_and_initialization_match(self):
        torch.manual_seed(2026);a=m.MomentNet(self.seq,self.ids,'static')
        torch.manual_seed(2026);b=m.MomentNet(self.seq,self.ids,'ordered')
        for (_,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):self.assertTrue(torch.equal(x,y))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))

    def test_temporal_controls(self):
        s=self.seq.reshape(24,3,2,15,6).mean(-1)
        shuffled=s[:,:,:,[5,2,7,0,14,8,4,1,10,6,12,9,3,13,11]]
        expanded=np.repeat(shuffled,6,-1)
        np.testing.assert_allclose(features(self.seq,'static'),features(expanded,'static'))
        np.testing.assert_allclose(features(self.seq,'sorted_marginals'),features(expanded,'sorted_marginals'))
        self.assertFalse(np.allclose(features(self.seq,'ordered_absolute'),features(expanded,'ordered_absolute')))

    def test_resume_bitwise(self):
        config={'seed':2026,'batch':64,'epochs':2,'min_epochs':2,'patience':8,'lr':.001,'min_lr':.00001,'weight_decay':.001}
        identity={'test':'resume_same_configuration'}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            m.fit(p/'full',config,identity,self.cache,self.y,self.u,1,'ordered')
            original=m.predict;calls=0
            def fail_after_saved_epoch(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('intentional interruption')
                return original(*args,**kwargs)
            with patch.object(m,'predict',side_effect=fail_after_saved_epoch):
                with self.assertRaisesRegex(RuntimeError,'intentional interruption'):
                    m.fit(p/'resume',config,identity,self.cache,self.y,self.u,1,'ordered')
            m.fit(p/'resume',config,identity,self.cache,self.y,self.u,1,'ordered')
            for name in ['best.pt','progress.pt']:
                a=torch.load(p/'full'/name,weights_only=False);b=torch.load(p/'resume'/name,weights_only=False)
                for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            a=np.load(p/'full/predictions.npz');b=np.load(p/'resume/predictions.npz')
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])


if __name__=='__main__':unittest.main()
