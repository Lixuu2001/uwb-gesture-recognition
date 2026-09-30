import sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as p

class ProtocolTests(unittest.TestCase):
    def test_subject_and_event_isolation(self):
        y=np.tile(np.repeat(np.arange(12),10),8); s=np.repeat(np.arange(1,9),120)
        tr,va,te=p.split(y,s,3,2026)
        self.assertEqual(set(s[te]),{3})
        self.assertNotIn(3,s[tr]); self.assertNotIn(3,s[va])
        self.assertFalse(set(tr)&set(va)); self.assertFalse(set(tr)&set(te))
        self.assertEqual(len(set(tr)|set(va)|set(te)),int(p.quality_mask(y,s).sum()))
        self.assertEqual(set(y[tr]),set(range(12)))
    def test_legacy_preprocessing_equivalence(self):
        from uwb_osr.data import MVDataset
        x=np.random.default_rng(1).normal(size=(2,3,90,189)).astype('float32')
        for mode in ('off','only','concat','signed','duplicate'):
            a=p.transform(x,mode)
            if mode in ('off','only','concat'):
                ds=MVDataset(x,np.array([0,1]),[0,1],normalize='abs_std',doppler=mode)
                np.testing.assert_allclose(a,np.stack([ds[i][0].numpy() for i in range(2)]),rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(p.transform(-x,'off'),p.transform(x,'off'))
    def test_phase_reversal_and_alias(self):
        t=np.arange(90)[None,:,None]; m=np.arange(189)[None,None,:]
        def rf(w): return np.cos(2*np.pi*40*m/189+w*t).astype('float32')
        a=p.phase_features(rf(.2))[0]; b=p.phase_features(rf(-.2))[0]
        self.assertGreater(a.mean(),0); self.assertLess(b.mean(),0)
        # > pi/frame aliases to negative: signed feature is NOT universal velocity truth.
        self.assertLess(p.phase_features(rf(3.5))[0].mean(),0)
    def test_no_test_in_preprocessor(self):
        x=np.random.default_rng(0).normal(size=(2,3,90,189)).astype('float32')
        a=p.transform(x[:1],'concat'); x[1]*=1e6
        np.testing.assert_array_equal(a,p.transform(x,'concat')[:1])
    def test_physics_dimension(self):
        x=np.random.default_rng(0).normal(size=(3,90,189)).astype('float32')
        c,q=p.phase_features(x)
        self.assertEqual(c.shape,(3,90)); self.assertEqual(q.shape,(3,90))
        self.assertEqual(p.motion_stats(c).shape,(3,13))
    def test_teacher_is_detached(self):
        from train import kd_loss
        s=torch.randn(5,12,requires_grad=True);t=torch.randn(5,12,requires_grad=True)
        kd_loss(s,t).backward()
        self.assertIsNone(t.grad);self.assertIsNotNone(s.grad)
    def test_concat_retains_view_order(self):
        from train import Classifier
        torch.set_num_threads(2);torch.manual_seed(0)
        x=torch.randn(1,3,90,189)
        with torch.no_grad():
            a=Classifier([0,1,2],'off','mean').eval()
            np.testing.assert_allclose(a(x)[0],a(x.flip(1))[0],atol=1e-5)
            b=Classifier([0,1,2],'off','concat').eval()
            self.assertGreater(float((b(x)[0]-b(x.flip(1))[0]).abs().max()),1e-4)

if __name__=='__main__': unittest.main()
