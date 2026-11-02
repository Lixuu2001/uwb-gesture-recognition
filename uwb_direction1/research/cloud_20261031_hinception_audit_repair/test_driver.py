"""Temporary metadata fixtures only; no real archive move or audit launch."""
import ast,collections,json,tempfile,unittest,os
from pathlib import Path
from repair_driver import preflight,sha
class Guards(unittest.TestCase):
 def fixture(self,tmp):
  b=Path(tmp)/'study';r=Path(tmp)/'repair';b.mkdir();r.mkdir();(b/'audit_supervisor01').mkdir();(b/'run01').mkdir()
  def put(p,v):p.write_text(json.dumps(v))
  (b/'science.py').write_text('# synthetic science metadata fixture\n');(b/'PROTOCOL.json').write_text('{}');put(b/'FROZEN_HASHES.json',{'science.py':sha(b/'science.py')});put(b/'run01/FILES.json',{});put(b/'audit_supervisor01/TERMINATION.json',dict(exit_code=1,wall_timeout=False,pid=2147483647));put(r/'ORIGINAL_FAILURE.json',dict(files_sha256={'science.py':sha(b/'science.py')}));put(r/'RNG_DIAGNOSIS.json',dict(restored_inspection_rng_then_selected_weights_equals_saved_final_without_real_forward=True));put(r/'REGRESSION_CHECK.json',dict(state='pass_synthetic_native_reload_observational_regression'));(r/'code.py').write_text('# fixture repair\n');put(r/'REPAIR_HASHES.json',{'code.py':sha(r/'code.py')});put(b/'AUDIT_REPAIR_REGISTRATION.json',dict(state='remote_manual_audit_repair_exact_readback_verified',protocol_sha256=sha(b/'PROTOCOL.json'),repair_manifest_sha256=sha(r/'REPAIR_HASHES.json')));return b,r
 def test_preflight_pure_and_missing_marker_refuses_without_moving(self):
  with tempfile.TemporaryDirectory() as d:
   b,r=self.fixture(d);self.assertTrue(preflight(b,r));(b/'AUDIT_REPAIR_REGISTRATION.json').unlink()
   with self.assertRaises(FileNotFoundError):preflight(b,r)
   self.assertTrue((b/'audit_supervisor01').exists());self.assertFalse((b/'audit_attempt01').exists())
 def test_modified_frozen_or_repair_or_registered_identity_refuses(self):
  for change in ['science','repair','marker']:
   with tempfile.TemporaryDirectory() as d:
    b,r=self.fixture(d)
    if change=='science':(b/'science.py').write_text('changed')
    elif change=='repair':(r/'code.py').write_text('changed')
    else:(b/'AUDIT_REPAIR_REGISTRATION.json').write_text('{}')
    with self.assertRaises(RuntimeError):preflight(b,r)
    self.assertTrue((b/'audit_supervisor01').exists())
 def test_existing_archive_completion_or_live_original_worker_refuses(self):
  for change in ['archive','complete','live']:
   with tempfile.TemporaryDirectory() as d:
    b,r=self.fixture(d)
    if change=='archive':(b/'audit_attempt01').mkdir()
    elif change=='complete':(b/'COMPLETE_AUDIT.json').write_text('{}')
    else:(b/'audit_supervisor01/TERMINATION.json').write_text(json.dumps(dict(exit_code=1,wall_timeout=False,pid=os.getpid())))
    with self.assertRaises(RuntimeError):preflight(b,r)
 def test_all_original_assertions_and_final_state_comparison_retained(self):
  original=ast.parse(Path('/workspace/uwb-hinception-pilot-20261031/audit_complete.py').read_bytes());repaired=ast.parse((Path(__file__).parent/'audit_complete_repaired.py').read_bytes())
  assertions=lambda t:collections.Counter(ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Assert))
  self.assertEqual(assertions(original),assertions(repaired))
  calls=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='compare']
  self.assertEqual(collections.Counter(calls(original)),collections.Counter(calls(repaired)))
if __name__=='__main__':unittest.main()
