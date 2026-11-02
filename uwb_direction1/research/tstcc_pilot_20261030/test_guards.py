import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_pilot as r
import audit_complete as a
class Guards(unittest.TestCase):
 def test_prior_ready_mutations(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);cfg=dict(reference_held1_macro=.9866666666666667);values={'previous_state':{'state':'complete','pid':None},'previous_summary':{'predeclared_route':'close_exact_configuration','reference_held1_macro':cfg['reference_held1_macro']},'previous_termination':{'exit_code':0,'wall_timeout':False},'previous_audit':{'state':'pass'},'previous_audit_receipt':{'state':'pass'},'previous_private':{'state':'private_readback_verified','modelset_recoverable':True}}
   for key,v in values.items():cfg[key]=str(b/key);(b/key).write_text(json.dumps(v))
   self.assertTrue(r.ready(cfg))
   for key,field,bad in [('previous_state','pid',123),('previous_state','state','training'),('previous_summary','predeclared_route','known_baseline_source_confirmation'),('previous_summary','reference_held1_macro',.5),('previous_termination','exit_code',1),('previous_audit','state','fail'),('previous_private','modelset_recoverable',False)]:
    altered=dict(values[key]);altered[field]=bad;(b/key).write_text(json.dumps(altered))
    with self.assertRaises(AssertionError):r.ready(cfg)
    (b/key).write_text(json.dumps(values[key]))
 def test_main_stops_before_arrays_or_models(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'PROTOCOL.json').write_text('{}')
   with patch.object(r,'B',b),patch.object(r,'ready',side_effect=RuntimeError('not_ready')),patch.object(r.np,'load',side_effect=AssertionError('array read forbidden')),patch.object(r,'make',side_effect=AssertionError('model forbidden')):
    with self.assertRaisesRegex(RuntimeError,'not_ready'):r.main()
   self.assertFalse((b/'run01').exists())
 def test_audit_active_and_failed_supervisor(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'run01').mkdir();s={'state':'training','pid':123};(b/'run01/STATE.json').write_text(json.dumps(s))
   with self.assertRaisesRegex(RuntimeError,'not_ready'):a.require_complete(b)
   s=dict(state='complete',pid=None,completed_phases=['ssl','linear','finetune','scratch'],all_predictions_frozen=True,diagnostic_labels_read=True,users7_8_labels_used=False);(b/'run01/STATE.json').write_text(json.dumps(s));(b/'supervisor01').mkdir();(b/'supervisor01/TERMINATION.json').write_text(json.dumps({'exit_code':1}))
   with self.assertRaises(AssertionError):a.require_complete(b)
 def test_macro_gate_and_roles(self):
  self.assertFalse(r.eligible({'linear':.9866666666666667},.9866666666666667));self.assertTrue(r.eligible({'finetune':.9916666666666667},.9866666666666667));self.assertFalse(r.eligible({'scratch':.9916666665},.9866666666666667))
  self.assertEqual({k:len(v) for k,v in r.roles(r.np.array([1,2,3,4,5,6])).items()},{'train':4,'validation':1,'diagnostic':1})
if __name__=='__main__':unittest.main(verbosity=2)
