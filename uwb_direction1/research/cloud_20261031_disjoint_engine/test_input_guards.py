"""Meaningful leakage/refusal boundary tests on synthetic metadata only."""
import unittest,tempfile,copy
from unittest.mock import patch
from input_guards import *
class Guards(unittest.TestCase):
 def setUp(self):
  self.expected={'training':[3,4,5,6],'validation':[2],'diagnostic':[1]}
  self.meta={r:dict(ids=np.arange(i*10,i*10+len(us),dtype='int64'),users=np.array(us,dtype='int64')) for i,(r,us) in enumerate(self.expected.items())}
 def test_valid_and_duplicate_roles(self):
  role_metadata(self.meta,self.expected)
  bad=copy.deepcopy(self.meta);bad['validation']['ids'][0]=bad['training']['ids'][0]
  with self.assertRaises(ValueError):role_metadata(bad,self.expected)
 def test_unallowed_user_and_overlap_and_wrong_user(self):
  for r,us in [('training',[3,4,7,6]),('validation',[1])]:
   ex=copy.deepcopy(self.expected);ex[r]=us
   with self.assertRaises(ValueError):role_metadata(self.meta,ex)
  bad=copy.deepcopy(self.meta);bad['training']['users'][0]=1
  with self.assertRaises(ValueError):role_metadata(bad,self.expected)
 def test_label_metadata_and_bad_ids(self):
  bad=copy.deepcopy(self.meta);bad['diagnostic']['labels']=np.array([0])
  with self.assertRaises(ValueError):role_metadata(bad,self.expected)
  for field,value in [('ids',np.array([1.0])),('ids',np.array([-1])),('users',np.array([True]))]:
   bad=copy.deepcopy(self.meta);bad['diagnostic'][field]=value
   with self.assertRaises(ValueError):role_metadata(bad,self.expected)
 def test_missing_current_audit_refuses_before_all_loaders(self):
  calls=[]
  def forbidden(*args):calls.append(args);raise AssertionError('Loader called before prior closes')
  with tempfile.TemporaryDirectory() as prior:
   with self.assertRaisesRegex(RuntimeError,'audit not complete'):load_registered_source('missing_protocol','missing_marker',prior,forbidden,forbidden,forbidden)
  self.assertEqual(calls,[])
 def test_only_train_val_labels_load_after_guard(self):
  # Inject a synthetic passing guard, never pretend current real audit passed.
  calls=[]
  with patch('input_guards.real_entry_ready',return_value={'role_users':self.expected}):
   out=load_registered_source('fixture','fixture','fixture',lambda:self.meta,lambda role,ids:('synthetic',role),lambda role,ids:calls.append(role))
  self.assertEqual(calls,['training','validation']);self.assertNotIn('diagnostic',out[3])
 def test_wrong_metadata_refuses_before_values_labels(self):
  bad=copy.deepcopy(self.meta);bad['validation']['ids']=bad['diagnostic']['ids'].copy();calls=[]
  with patch('input_guards.real_entry_ready',return_value={'role_users':self.expected}):
   with self.assertRaises(ValueError):load_registered_source('fixture','fixture','fixture',lambda:bad,lambda *a:calls.append(a),lambda *a:calls.append(a))
  self.assertEqual(calls,[])
if __name__=='__main__':unittest.main()
