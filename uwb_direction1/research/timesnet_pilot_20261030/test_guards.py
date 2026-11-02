import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_pilot as r
import audit_complete as a
from ops import training_order,evaluation_order
class Guards(unittest.TestCase):
 def test_prior_guard_mutations(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cfg={'reference_held1_macro':.9866666666666667};items={'previous_state':{'state':'complete','pid':None},'previous_summary':{'predeclared_route':'close_exact_configuration','reference_held1_macro':cfg['reference_held1_macro']},'previous_termination':{'exit_code':0,'wall_timeout':False},'previous_audit_termination':{'exit_code':0,'wall_timeout':False},'previous_audit':{'state':'pass'},'previous_private':{'state':'private_readback_verified','all_completed_models_included':True,'standalone_complete_modelset_recoverable':True}}
   for key,v in items.items():cfg[key]=str(p/key);(p/key).write_text(json.dumps(v))
   self.assertTrue(r.ready(cfg))
   for key,field,bad in [('previous_state','pid',7),('previous_state','state','training'),('previous_summary','reference_held1_macro',.5),('previous_summary','predeclared_route','known_baseline_source_confirmation'),('previous_audit_termination','exit_code',1),('previous_private','all_completed_models_included',False),('previous_private','standalone_complete_modelset_recoverable',False)]:
    v=dict(items[key]);v[field]=bad;(p/key).write_text(json.dumps(v))
    with self.assertRaises(AssertionError):r.ready(cfg)
    (p/key).write_text(json.dumps(items[key]))
 def test_actual_main_guard_before_arrays_models(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'PROTOCOL.json').write_text('{}')
   with patch.object(r,'B',p),patch.object(r,'ready',side_effect=RuntimeError('not_ready')),patch.object(r.np,'load',side_effect=AssertionError('array forbidden')),patch.object(r,'make',side_effect=AssertionError('model forbidden')):
    with self.assertRaisesRegex(RuntimeError,'not_ready'):r.main()
   self.assertFalse((p/'run01').exists())
 def test_complete_guard_rejects_active_or_failed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'run01').mkdir();(p/'run01/STATE.json').write_text(json.dumps({'state':'training','pid':7}))
   with self.assertRaisesRegex(RuntimeError,'not_ready'):a.require_complete(p)
   (p/'run01/STATE.json').write_text(json.dumps({'state':'complete','pid':None,'all_predictions_frozen':True,'diagnostic_labels_read':True,'users7_8_labels_used':False}));(p/'supervisor01').mkdir();(p/'supervisor01/TERMINATION.json').write_text(json.dumps({'exit_code':1}))
   with self.assertRaises(AssertionError):a.require_complete(p)
 def test_labelblind_partitions_and_last_batch(self):
  for n,role,offset in [(1000,'validation',101),(1200,'diagnostic',102)]:
   order=evaluation_order(n,role);r.np.testing.assert_array_equal(order,r.np.random.RandomState(20261029+offset).permutation(n));self.assertEqual(len(r.np.unique(order)),n)
  for epoch in [1,2,30]:
   order=training_order(epoch);self.assertEqual(len(r.np.unique(order)),4700);batches=[order[i:i+16] for i in range(0,4700,16)];self.assertEqual(len(batches),294);self.assertEqual(len(batches[-1]),12)
 def test_original_prediction_rule_and_gate(self):
  x=r.np.array([[1.,2.,0.],[3.,1.,2.]],dtype=r.np.float32);r.np.testing.assert_array_equal(r.predictions(x),[1,0]);self.assertTrue(r.gate(.9916666666666667,.9866666666666667));self.assertFalse(r.gate(.9916666665,.9866666666666667))
if __name__=='__main__':unittest.main(verbosity=2)
