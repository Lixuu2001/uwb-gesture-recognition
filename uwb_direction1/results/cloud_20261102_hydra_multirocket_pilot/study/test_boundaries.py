from pathlib import Path
import sys,pickle,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,'/workspace/uwb-hydra-multirocket-source-probe-20261102')
from native_hydramr import np,make,assert_native,components
from engine import predict,fitted_digest
from audit_complete import verify_scaler_statistics,arrays
from metrics import gate
import finalize
from common import ready
from independent_supervisor import bounded_run
class ProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.RandomState(20261029).randn(24,567,90).astype('float32');cls.x[:,:16]=0
        cls.m=make();cls.m.fit(cls.x,np.arange(24)%12);assert_native(cls.m,True)
    def test_native_staged_immutable_and_reload(self):
        before=fitted_digest(self.m);maps=self.x.transpose(0,2,1)
        result=predict(self.m,maps,np.arange(24),np.arange(24));np.testing.assert_array_equal(result['predictions'],self.m.predict(self.x))
        self.assertEqual(before,fitted_digest(self.m));clone=pickle.loads(pickle.dumps(self.m,protocol=5))
        self.assertEqual(before,fitted_digest(clone));np.testing.assert_array_equal(result['predictions'],clone.predict(self.x))
        for a,b in zip(arrays(self.m,np),arrays(clone,np)):np.testing.assert_array_equal(a,b)
    def test_streaming_independent_statistics(self):
        sums=np.zeros(4096);sq=sums.copy();zero=np.zeros(4096,np.int64);ms=np.zeros(49728);msq=ms.copy()
        for offset in range(0,24,7):
            h,mr,_=components(self.m,self.x[offset:offset+7]);q=np.sqrt(np.maximum(h.numpy().astype(np.float64),0));d=mr.astype(np.float64)
            sums+=q.sum(0);sq+=(q*q).sum(0);zero+=(q==0).sum(0);ms+=d.sum(0);msq+=(d*d).sum(0)
        verify_scaler_statistics(np,self.m,sums,sq,zero,ms,msq,24)
    def test_scientific_tensor_tampering_detected(self):
        clone=pickle.loads(pickle.dumps(self.m,protocol=5));before=fitted_digest(clone)
        clone._transform_hydra._hydra.W[0,0,0,0,0]+=.001;self.assertNotEqual(before,fitted_digest(clone))
    def test_fixed_gate_and_support(self):
        self.assertFalse(gate(.9866666666666667)['predeclared_gate_passed']);self.assertTrue(gate(.9916666666666667)['predeclared_gate_passed'])
        for n in [10,12]:
            y=np.repeat(np.arange(n),10);p=y.copy();p[:5]=n-1
            expected=float(np.mean([np.mean(p[y==k]==k) for k in range(n)]))
            self.assertEqual(finalize.supported_macro(np,y,p,range(n)),expected)
        with self.assertRaises(AssertionError):finalize.supported_macro(np,np.arange(10),np.arange(10),range(12))
    def test_no_remote_marker_or_private_recovery_blocks_access(self):
        with self.assertRaises(FileNotFoundError):ready()
        with patch.object(finalize,'complete',return_value={}):
            with self.assertRaises(FileNotFoundError):finalize.main()
    def test_supervisor_keeps_failed_and_timeout_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            a=bounded_run([sys.executable,'-c','raise SystemExit(3)'],root/'failed',5,3,4)
            self.assertEqual(a['exit_code'],3);self.assertFalse(a['scientific_negative']);self.assertTrue((root/'failed/TERMINATION.json').is_file())
            a=bounded_run([sys.executable,'-c','import time;time.sleep(10)'],root/'timeout',1,3,4)
            self.assertTrue(a['wall_timeout']);self.assertFalse(a['scientific_negative']);self.assertTrue((root/'timeout/TERMINATION.json').is_file())
if __name__=='__main__':unittest.main()
