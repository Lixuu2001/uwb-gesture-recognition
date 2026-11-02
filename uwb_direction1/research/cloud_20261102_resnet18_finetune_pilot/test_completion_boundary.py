import unittest,tempfile,json
from pathlib import Path
import common
class Completion(unittest.TestCase):
 def test_missing_remote_registration_blocks_before_inputs(self):
  old=common.B
  with tempfile.TemporaryDirectory() as d:
   common.B=Path(d)
   try:
    with self.assertRaises(FileNotFoundError):common.ready()
   finally:common.B=old
 def test_resource_failure_or_partial_full_replay_cannot_unlock_labels(self):
  oldb,oldready=common.B,common.ready
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);common.B=b;common.ready=lambda:{}
   try:
    term=dict(exit_code=0,wall_timeout=False,resource_stop_reason=None,cgroup_oom_kill_delta=0)
    for n in ['supervisor01','audit_supervisor01']:(b/n).mkdir();(b/n/'TERMINATION.json').write_text(json.dumps(term))
    fields=['full40_epoch12560_update_replay_exact','all40_validation_scores_selection_exact','best_full_model_and_final_model_optimizer_momentum_rng_exact','all2200_selected_scores_predictions_exact','all6900_raw_RMS_cache_FP64_verified'];a=dict(state='pass',diagnostic_labels_read=False,users7_8_labels_used=False,artifact_hashes={},**{k:True for k in fields});a[fields[0]]=False;(b/'COMPLETE_AUDIT.json').write_text(json.dumps(a))
    with self.assertRaises(AssertionError):common.complete()
    a[fields[0]]=True;(b/'COMPLETE_AUDIT.json').write_text(json.dumps(a));term['resource_stop_reason']='RSS_resource_limit';(b/'audit_supervisor01/TERMINATION.json').write_text(json.dumps(term))
    with self.assertRaises(AssertionError):common.complete()
   finally:common.B,common.ready=oldb,oldready
if __name__=='__main__':unittest.main(verbosity=2)
