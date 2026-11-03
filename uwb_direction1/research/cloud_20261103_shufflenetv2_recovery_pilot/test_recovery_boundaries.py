"""Recovery-only structural fixtures; no UWB arrays or actual workers."""
import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import common,controller
FIELDS=['full40_epoch12560_update_replay_exact','all40_validation_scores_selection_exact','best_full_model_and_final_model_optimizer_momentum_rng_exact','all2200_selected_scores_predictions_exact','all6900_raw_RMS_cache_FP64_verified']
class RecoveryBoundaries(unittest.TestCase):
 def test_absent_initial_private_insurance_prevents_worker_launch(self):
  with tempfile.TemporaryDirectory() as d,patch.object(controller,'B',Path(d)),patch.object(controller,'ready',return_value={}),patch.object(controller,'bounded_run') as spawn:
   with self.assertRaises(FileNotFoundError):controller.main()
   spawn.assert_not_called()
 def test_mismatched_insurance_freeze_prevents_worker_launch(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'INITIAL_PRIVATE_INSURANCE.json').write_text(json.dumps(dict(state='owner_only_raw_allmember_initial_insurance_verified',frozen_sha256='incorrect',diagnostic_labels_scored=False,shared=False)));(b/'FROZEN_HASHES.json').write_text('{}')
   with patch.object(controller,'B',b),patch.object(controller,'ready',return_value={}),patch.object(controller,'bounded_run') as spawn:
    with self.assertRaises(AssertionError):controller.main()
    spawn.assert_not_called()
 def fixture(self,b):
  term=dict(exit_code=0,wall_timeout=False,resource_stop_reason=None,cgroup_oom_kill_delta=0)
  for n in ['supervisor01','audit_supervisor01']:(b/n).mkdir();(b/n/'TERMINATION.json').write_text(json.dumps(term))
  return dict(state='pass',diagnostic_labels_read=False,users7_8_labels_used=False,pixel_count=1038643200,artifact_hashes={},**{k:True for k in FIELDS})
 def test_complete_flags_with_empty_artifact_hash_loop_cannot_pass(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);a=self.fixture(b);(b/'COMPLETE_AUDIT.json').write_text(json.dumps(a))
   with patch.object(common,'B',b),patch.object(common,'ready',return_value={}):
    with self.assertRaises(AssertionError):common.complete()
 def test_valid_virtual_model_map_then_missing_model_or_outside_path(self):
  with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as outside:
   b=Path(d);a=self.fixture(b);run=b/'run01';run.mkdir();names=['initial12_head.pt','best_model.pt','final_full.pt','split.npz','steps.jsonl','TRAIN_SELECTION.json','FILES.json','validation_unscored.npz','diagnostic_unscored.npz']+[f'validation_epoch{i:02d}.npz' for i in range(1,41)]
   for n in names:p=run/n;p.write_bytes(b'synthetic');a['artifact_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
   def save():(b/'COMPLETE_AUDIT.json').write_text(json.dumps(a))
   save()
   with patch.object(common,'B',b),patch.object(common,'ready',return_value={}):
    self.assertEqual(common.complete(),{})
    digest=a['artifact_hashes'].pop(str(run/'final_full.pt'));save()
    with self.assertRaises(AssertionError):common.complete()
    a['artifact_hashes'][str(run/'final_full.pt')]=digest
    p=Path(outside)/'not_this_study';p.write_bytes(b'x');a['artifact_hashes'][str(p)]=hashlib.sha256(b'x').hexdigest();save()
    with self.assertRaises(AssertionError):common.complete()
if __name__=='__main__':unittest.main(verbosity=2)
