"""Guard checks have no real data loader or model execution."""
import copy,tempfile,unittest
from pathlib import Path
from guards import fixed,read,ready
B=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_current_fixed_metadata(self):fixed(read(B/'PROTOCOL.json'))
 def test_configuration_refusal(self):
  p=read(B/'PROTOCOL.json')
  for k,v in [('epochs',99),('seed',20261030),('batch_size',64),('users7_8_labels_used',True),('cost_gate_passed',False),('max_training_updates',9000),('input_shape',[567,90]),('loss','mean_squared_error'),('training_unweighted',False),('source_callbacks_only_min_train_loss',False),('reference_held1_macro',0.5)]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):fixed(q)
 def test_role_and_gate_refusal(self):
  p=read(B/'PROTOCOL.json')
  for k,v in [('training_users',[3,4,5,7]),('diagnostic_user',8),('validation_user',1),('counts',{'train':4700,'validation':1200,'diagnostic':1000}),('gate',{})]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):fixed(q)
 def test_missing_registration_before_engine_import(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'PROTOCOL.json').write_bytes((B/'PROTOCOL.json').read_bytes())
   with self.assertRaises(FileNotFoundError):ready(Path(d))
  import sys;self.assertNotIn('engine',sys.modules)
if __name__=='__main__':unittest.main()
