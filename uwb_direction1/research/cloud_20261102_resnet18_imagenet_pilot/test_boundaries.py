import unittest,tempfile,pickle,copy,sys
from pathlib import Path
from engine import np,RoleMaps,pixels_reference,preprocess,make_head,assert_head,payload,assert_payload,private_scope,WEIGHTS_SHA,SCOPE
from metrics import gate
from finalize import supported_macro
class Checks(unittest.TestCase):
 def test_role_axis_mapping_and_FP64_pixels(self):
  rng=np.random.default_rng(20261029);raw=rng.normal(size=(3,3,90,189)).astype(np.float32);raw[0,0]=1;maps=raw.transpose(0,2,1,3).reshape(3,90,567);provider=RoleMaps(maps,np.array([2,0]));x=provider[:2];np.testing.assert_array_equal(x,raw[[2,0]]);np.testing.assert_allclose(preprocess(x).numpy(),pixels_reference(x),rtol=2e-5,atol=2e-5)
 def test_fixed_SVM_deterministic_refit_restore_and_tamper(self):
  rng=np.random.default_rng(20261029);f=rng.normal(size=(72,512));y=np.arange(72)%12;a=make_head().fit(f,y);b=make_head().fit(f,y);assert_head(a,True)
  for key in ['coef_','intercept_','classes_','n_iter_']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
  q=pickle.loads(pickle.dumps(payload(a),protocol=5));assert_payload(q,True);np.testing.assert_array_equal(q['head'].decision_function(f),a.decision_function(f))
  q['encoder_weights_sha256']='bad'
  with self.assertRaises(AssertionError):assert_payload(q,True)
  q=payload(copy.deepcopy(a));q['head'].coef_[0,0]=np.nan
  with self.assertRaises(AssertionError):assert_payload(q,True)
 def test_known_support_and_fixed_gate(self):
  y=np.repeat(np.arange(10),2);self.assertEqual(supported_macro(np,y,y,range(10)),1.)
  with self.assertRaises(AssertionError):supported_macro(np,y,y,range(12))
  self.assertFalse(gate(.9866666666666667,.9866666666666667,.005,1e-12)['predeclared_gate_passed']);self.assertTrue(gate(.995,.9866666666666667,.005,1e-12)['predeclared_gate_passed'])
 def test_head_only_or_missing_private_backup_blocks_score(self):
  m=dict(scope=SCOPE,complete_modelset=True,shared_frozen_encoder_included_once=True,archive_sha256='same');v=dict(scope=SCOPE,complete_modelset_verified=True,archive_sha256='same');r=dict(scope=SCOPE,complete_modelset_verified=True,numeric_audit_certified=True,state='private_readback_verified',archive_sha256='same');private_scope(m,v,r,{'source_preparation/weights/resnet18-f37072fd.pth':WEIGHTS_SHA})
  with self.assertRaises((AssertionError,KeyError)):private_scope(m,v,r,{})
  with self.assertRaises(AssertionError):private_scope(dict(m,shared_frozen_encoder_included_once=False),v,r,{'source_preparation/weights/resnet18-f37072fd.pth':WEIGHTS_SHA})
 def test_missing_remote_marker_blocks_before_data(self):
  import common
  previous=common.B
  try:
   with tempfile.TemporaryDirectory() as d:
    common.B=Path(d)
    with self.assertRaises(FileNotFoundError):common.ready()
  finally:common.B=previous
if __name__=='__main__':unittest.main(verbosity=2)
