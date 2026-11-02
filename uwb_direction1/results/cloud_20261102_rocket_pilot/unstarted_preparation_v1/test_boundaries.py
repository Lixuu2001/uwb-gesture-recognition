import unittest,sys
from pathlib import Path
from metrics import gate
class ProtocolBoundaries(unittest.TestCase):
    def test_fixed_gate(self):
        self.assertFalse(gate(.9866666666666667)['predeclared_gate_passed'])
        self.assertTrue(gate(.9916666666666667)['predeclared_gate_passed'])
        self.assertFalse(gate(.99)['predeclared_gate_passed'])
    def test_finite(self):
        for v in [float('nan'),float('inf'),-1.,2.]:
            with self.assertRaises(AssertionError):gate(v)
    def test_native_staged_predict_equivalence(self):
        sys.path.insert(0,'/workspace/uwb-rocket-source-probe-20261102')
        from native_rocket import make,np
        from engine import predict
        g=np.random.RandomState(20261029);x=g.randn(24,90,567).astype('float32');m=make();m.fit(x.transpose(0,2,1),np.arange(24)%12)
        result=predict(m,x,np.arange(24),np.arange(24));np.testing.assert_array_equal(result['predictions'],m.predict(x.transpose(0,2,1)))
        np.testing.assert_array_equal(result['decisions'],m.estimator_.decision_function(m.pipeline_[:-1].transform(x.transpose(0,2,1))))
    def test_no_diagnostic_label_loader_in_producer_or_auditor(self):
        import ast
        for name in ['run_pilot.py','audit_complete.py']:
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Constant) and isinstance(n.value,str):self.assertNotIn('y_clutterremoved.npy',n.value)
if __name__=='__main__':unittest.main()
