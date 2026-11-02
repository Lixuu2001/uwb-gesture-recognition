"""Guards must refuse unfinished/partial work before diagnostic-summary access."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import readiness_guards as guard
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def fixture(p):
 audit=dict(state='pass',training_updates_replayed_exact=29400,epochs_replayed=100,selection='source_final_last',users7_8_labels_used=False,known_baseline_expansion_eligible=False);put(p/'COMPLETE_AUDIT.json',audit)
 for n in ['supervisor01/TERMINATION.json','audit_supervisor01/TERMINATION.json']:put(p/n,dict(exit_code=0,wall_timeout=False))
 b=p/'complete_backup_20261031';put(b/'payload/MEMBERS.json',dict(members=[dict(path='study/COMPLETE_AUDIT.json',sha256=guard.sha(p/'COMPLETE_AUDIT.json'))]));put(b/'MANIFEST.json',dict(archive_sha256='synthetic_archive_sha_only',members_manifest_sha256=guard.sha(b/'payload/MEMBERS.json'),tracked_payload_files=1));put(b/'READBACK_VERIFIED.json',dict(state='remote_archive_and_all_members_verified',archive_sha256='synthetic_archive_sha_only',tracked_payload_members_verified=1));put(b/'DRIVE_BACKUP_RECEIPT.json',dict(state='private_readback_verified',standalone_complete_modelset_recoverable=True,all_chunks_raw_authenticated_readback_and_members_verified=True,all_completed_models_included=True,archive_sha256='synthetic_archive_sha_only'));put(p/'run01/SUMMARY.json',dict(predeclared_route='close_exact_configuration',known_baseline_expansion_eligible=False))
class Guards(unittest.TestCase):
 def test_missing_audit_refuses_before_any_json_read(self):
  with tempfile.TemporaryDirectory() as d,patch.object(guard,'read',side_effect=AssertionError('Must not read diagnostic summary')):
   with self.assertRaisesRegex(RuntimeError,'not complete'):guard.prior_closed(d)
 def test_failed_or_partial_audit_refuses_before_summary(self):
  for a in [dict(state='incomplete'),dict(state='pass',training_updates_replayed_exact=1)]:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);put(p/'COMPLETE_AUDIT.json',a)
    with self.assertRaisesRegex(RuntimeError,'not passing|partial'):guard.prior_closed(p)
 def test_failed_termination_refuses_before_backup_summary(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fixture(p);put(p/'audit_supervisor01/TERMINATION.json',dict(exit_code=1,wall_timeout=False));(p/'run01/SUMMARY.json').unlink()
   with self.assertRaisesRegex(RuntimeError,'execution incomplete'):guard.prior_closed(p)
 def test_first_or_partial_insurance_is_not_complete(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fixture(p);put(p/'complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json',dict(state='private_readback_verified',all_completed_models_included=False));(p/'run01/SUMMARY.json').unlink()
   with self.assertRaisesRegex(RuntimeError,'complete private'):guard.prior_closed(p)
 def test_mismatched_backup_digest_or_audit_coverage_refuses(self):
  for bad in ['digest','coverage']:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);fixture(p);b=p/'complete_backup_20261031'
    if bad=='digest':put(b/'READBACK_VERIFIED.json',dict(state='remote_archive_and_all_members_verified',archive_sha256='wrong',tracked_payload_members_verified=1))
    else:
     put(b/'payload/MEMBERS.json',dict(members=[]));m=guard.read(b/'MANIFEST.json');m['members_manifest_sha256']=guard.sha(b/'payload/MEMBERS.json');put(b/'MANIFEST.json',m)
    (p/'run01/SUMMARY.json').unlink()
    with self.assertRaisesRegex(RuntimeError,'inconsistent|absent'):guard.prior_closed(p)
 def test_prior_pass_demands_source_confirmation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fixture(p);put(p/'run01/SUMMARY.json',dict(predeclared_route='known_baseline_source_confirmation',known_baseline_expansion_eligible=True))
   with self.assertRaisesRegex(RuntimeError,'source confirmation'):guard.prior_closed(p)
 def test_empty_configuration_and_wrong_marker_refuse(self):
  with tempfile.TemporaryDirectory() as d,patch.object(guard,'prior_closed',return_value=True):
   p=Path(d);put(p/'protocol.json',dict(seed=None,epochs=None,batch_size=None))
   with self.assertRaisesRegex(RuntimeError,'Unset'):guard.real_entry_ready(p/'protocol.json',p/'absent',p)
   put(p/'protocol.json',dict(seed=7,epochs=3,batch_size=4,variant='author_LITEMV_full',input_preprocessing='explicit_RMS_adaptation',cost_gate_passed=True));put(p/'marker.json',dict(state='remote_preregistration_exact_readback_verified',protocol_sha256='wrong'))
   with self.assertRaisesRegex(RuntimeError,'preregistration'):guard.real_entry_ready(p/'protocol.json',p/'marker.json',p)
 def test_valid_synthetic_closed_fixture(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);fixture(p);self.assertTrue(guard.prior_closed(p))
if __name__=='__main__':unittest.main()
