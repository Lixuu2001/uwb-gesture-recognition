import copy,unittest
from observe import completed
class Completion(unittest.TestCase):
 def setUp(self):
  self.c=dict(max_training_updates=39200,epochs=100);self.s=dict(state='complete_audited_unscored_requires_private_backup_and_finalization',pid=None)
  self.train=dict(state='complete',pid=None,all_predictions_frozen=True);self.term=dict(exit_code=0,wall_timeout=False)
  self.cert=dict(state='pass',updates_replayed=39200,epochs_replayed=100,diagnostic_labels_read=False)
 def call(self,at=None,cert=None,active=None):return completed(self.c,self.s,self.train,self.term,at or self.term,cert or self.cert,active or [])
 def test_completed_without_active_process(self):self.assertTrue(self.call())
 def test_failed_or_missing_audit_rejected(self):
  with self.assertRaises(RuntimeError):self.call(at=dict(exit_code=-9,wall_timeout=True))
  with self.assertRaises(RuntimeError):completed(self.c,self.s,self.train,self.term,None,None,[])
 def test_partial_or_active_rejected(self):
  bad=copy.deepcopy(self.cert);bad['updates_replayed']=1
  with self.assertRaises(RuntimeError):self.call(cert=bad)
  with self.assertRaises(RuntimeError):self.call(active=[dict(active=True)])
if __name__=='__main__':unittest.main()
