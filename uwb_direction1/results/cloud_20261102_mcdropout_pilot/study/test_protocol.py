from pathlib import Path
import copy,json,tempfile,unittest
from guards import fixed,ready,read,require_inference_complete
B=Path(__file__).resolve().parent
class Protocol(unittest.TestCase):
 def test_fixed(self):fixed(read(B/'PROTOCOL.json'))
 def test_policy_refusal(self):
  p=read(B/'PROTOCOL.json')
  for k,v in [('samples',99),('batch_size',32),('seed',7),('dropout_rate',.1),('dropout_nodes',2),('link','softmax'),('new_training',True),('users7_8_labels_used',True),('selected_epoch',42),('diagnostic_user',8),('validation_user',1),('reused_training_users',[3,4,5,7]),('role_seed_offsets',{}),('gate',{}),('reference_held1_macro',.5),('cost_gate_passed',False)]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):fixed(q)
 def test_missing_registration_before_checkpoint_values(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'PROTOCOL.json').write_bytes((B/'PROTOCOL.json').read_bytes())
   with self.assertRaises(FileNotFoundError):ready(d)
 def test_running_inference_refused(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'run01').mkdir();(b/'supervisor01').mkdir();(b/'run01/STATE.json').write_text(json.dumps(dict(state='inference',pid=1,diagnostic_labels_read=False,users7_8_labels_used=False)));(b/'supervisor01/TERMINATION.json').write_text(json.dumps(dict(exit_code=0,wall_timeout=False)))
   with self.assertRaises(RuntimeError):require_inference_complete(b)
if __name__=='__main__':unittest.main()
