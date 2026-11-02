from pathlib import Path
import sys,pickle,tempfile,unittest
from unittest.mock import patch
from engine import np,make,assert_native,predict,fitted_digest,scientific_arrays
from audit_complete import verify_statistics,arrays
from aeon.transformations.collection import Normalizer
from common import ready
from metrics import gate
from independent_supervisor import bounded_run
import finalize
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.RandomState(20261029);y=np.arange(24)%12;phase=np.linspace(0,2*np.pi,90,endpoint=False,dtype=np.float32)
        cls.x=(.15*rng.randn(24,567,90)+3*np.sin((y[:,None,None]+1)*phase[None,None,:])).astype('float32');cls.x[:,0]=0
        cls.m=make();cls.m.fit(cls.x,y);assert_native(cls.m,True)
    def test_native_full_probabilities_votes_canonical_predictions_reload_immutable(self):
        before=fitted_digest(self.m);r=predict(self.m,self.x.transpose(0,2,1),np.arange(24),np.arange(24))
        np.testing.assert_array_equal(r['probabilities'],self.m.predict_proba(self.x));np.testing.assert_array_equal(r['predictions'],self.m.predict(self.x));self.assertEqual(before,fitted_digest(self.m))
        clone=pickle.loads(pickle.dumps(self.m,protocol=5));self.assertEqual(before,fitted_digest(clone))
        for a,b in zip(scientific_arrays(self.m),arrays(clone,np)):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(clone.predict_proba(self.x),r['probabilities'])
    def test_independent_all25_streaming_stats(self):
        total=np.zeros((25,4000));sq=total.copy()
        for offset in range(0,24,7):
            norm=Normalizer().fit_transform(self.x[offset:offset+7]).astype('float32',copy=False)
            for i,pipe in enumerate(self.m.estimators_):
                f=pipe.steps[0][1]._transform_kernels(norm).astype(np.float64);total[i]+=f.sum(0);sq[i]+=(f*f).sum(0)
        verify_statistics(np,[p.steps[1][1] for p in self.m.estimators_],total,sq,24)
    def test_kernel_and_vote_tampering_detected(self):
        m=pickle.loads(pickle.dumps(self.m,protocol=5));before=fitted_digest(m);m.estimators_[0].steps[0][1].kernels[0][0]+=.001;self.assertNotEqual(before,fitted_digest(m))
        m=pickle.loads(pickle.dumps(self.m,protocol=5));m.weights_[0]+=.001;self.assertNotEqual(before,fitted_digest(m))
        m=pickle.loads(pickle.dumps(self.m,protocol=5));m.estimators_[0].steps[2][1].coef_[0,0]=np.nan
        with self.assertRaises(AssertionError):fitted_digest(m)
    def test_nontrivial_native_training_LOO_accuracy_alpha_and_weight(self):
        from aeon.classification.convolution_based._arsenal import _fit_ridge_classifier
        rng=np.random.RandomState(20261029);y=np.arange(72)%12
        z=rng.randn(12,16)[y]+.8*rng.randn(72,16);z-=z.mean(0)
        r=_fit_ridge_classifier(z,y,None)
        Y=2*(y[:,None]==np.arange(12)).astype(float)-1;K=z@z.T
        Q=np.linalg.qr(np.vstack([np.eye(71),-np.ones((1,71))]))[0];scores=[]
        for alpha in r.alphas:
            A=Q@np.linalg.inv(Q.T@K@Q+alpha*np.eye(71))@Q.T
            error=(A@(Y-Y.mean(0)))/np.diag(A)[:,None]
            scores.append(float(np.mean(np.argmax(Y-error,axis=1)==y)))
        self.assertGreater(len(set(scores)),1)
        self.assertEqual(float(r.best_score_),max(scores))
        self.assertEqual(scores[list(r.alphas).index(r.alpha_)],max(scores))
    def test_gate_and_known_support(self):
        self.assertFalse(gate(.9866666666666667)['predeclared_gate_passed']);self.assertTrue(gate(.9916666666666667)['predeclared_gate_passed'])
        for n in [10,12]:self.assertEqual(finalize.supported_macro(np,np.arange(n),np.arange(n),range(n)),1)
        with self.assertRaises(AssertionError):finalize.supported_macro(np,np.arange(10),np.arange(10),range(12))
    def test_missing_remote_and_private_proofs_block(self):
        with self.assertRaises(FileNotFoundError):ready()
        with patch.object(finalize,'complete',return_value={}):
            with self.assertRaises(FileNotFoundError):finalize.main()
    def test_failed_timeout_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);a=bounded_run([sys.executable,'-c','raise SystemExit(3)'],root/'failed',5,3,4);self.assertEqual(a['exit_code'],3);self.assertFalse(a['scientific_negative'])
            a=bounded_run([sys.executable,'-c','import time;time.sleep(10)'],root/'timeout',1,3,4);self.assertTrue(a['wall_timeout']);self.assertTrue((root/'timeout/TERMINATION.json').is_file())
if __name__=='__main__':unittest.main(verbosity=2)
