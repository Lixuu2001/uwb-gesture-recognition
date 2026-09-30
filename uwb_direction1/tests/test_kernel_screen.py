import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from kernel_screen import source_map, kernel, features, fit, weights_for

class KernelTests(unittest.TestCase):
    def test_psd_and_unit_diagonal(self):
        rng=np.random.default_rng(5); x=rng.normal(size=(25,3,4))
        for kind in ['additive','joint']:
            k=kernel(x,x,np.ones(3),kind)
            np.testing.assert_allclose(k,k.T,atol=1e-14)
            np.testing.assert_allclose(np.diag(k),1,atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-10)

    def test_missing_cells_balanced(self):
        y=np.array([0,0,0,1,1]); u=np.array([1,1,2,1,1])
        w=weights_for(y,u)
        self.assertAlmostEqual(w.sum(),1)
        self.assertAlmostEqual(w[y==0].sum(),.5)
        self.assertAlmostEqual(w[(y==0)&(u==1)].sum(),.25)

    def test_source_mapping_does_not_change_on_inference(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(48,3,4))
        y=np.tile(np.arange(12),4); u=np.repeat(np.arange(4),12)
        m=source_map(x,y,u,2026,24); before={k:v.copy() for k,v in m.items()}
        features(rng.normal(size=(4,3,4))*1e6,m,'joint',np.eye(24))
        for k in m: np.testing.assert_array_equal(m[k],before[k])
        self.assertEqual(len(np.unique(m['landmark_indices'])),24)
        again=source_map(x,y,u,2026,24)
        for k in m: np.testing.assert_array_equal(m[k],again[k])

    def test_interaction_xor_distinguishes_additive_control(self):
        # A balanced XOR has no additive main effect, but a product RBF represents it.
        x=np.array([[[a],[b],[0]] for a in [-1,1] for b in [-1,1]],dtype=float)
        y=np.array([0,1,1,0]); w=np.ones(4)/4
        outcomes={}
        for kind in ['additive','joint']:
            k=kernel(x,x,np.ones(3),kind)
            v,q=np.linalg.eigh(k); phi=k@(q/np.sqrt(np.maximum(v,1e-10)))
            coef,bias=fit(phi,y,w,1e-5); scores=phi@coef+bias
            outcomes[kind]=scores
        np.testing.assert_allclose(outcomes['additive'][:,0],outcomes['additive'][:,1],atol=1e-9)
        np.testing.assert_array_equal(outcomes['joint'].argmax(-1),y)

if __name__=='__main__': unittest.main()
