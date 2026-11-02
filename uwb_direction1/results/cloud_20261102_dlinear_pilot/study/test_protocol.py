import copy,tempfile,unittest
from pathlib import Path
from guards import read,fixed,ready
B=Path(__file__).resolve().parent
class Protocol(unittest.TestCase):
 def test_current(self):fixed(read(B/'PROTOCOL.json'))
 def test_policy_refusal(self):
  p=read(B/'PROTOCOL.json')
  for k,v in [('epochs',9),('batch_size',64),('seed',7),('train_shuffle',False),('users7_8_labels_used',True),('patience',4),('optimizer','Adam.001'),('gradient_clip',1.),('loss','NLLLoss'),('input_shape',[567,90]),('max_training_updates',1),('cost_gate_passed',False),('reference_held1_macro',.5)]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):fixed(q)
 def test_role_gate_refusal(self):
  p=read(B/'PROTOCOL.json')
  for k,v in [('training_users',[3,4,5,7]),('diagnostic_user',8),('validation_user',1),('counts',{}),('gate',{})]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):fixed(q)
 def test_absent_registration_before_data(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'PROTOCOL.json').write_bytes((B/'PROTOCOL.json').read_bytes())
   with self.assertRaises(FileNotFoundError):ready(d)
if __name__=='__main__':unittest.main()
