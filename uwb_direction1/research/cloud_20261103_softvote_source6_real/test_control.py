"""Synthetic control fixtures only; never claims real-model full100 fitting."""
import unittest,tempfile,os,json
from pathlib import Path
from unittest.mock import patch
import common,controller
class Control(unittest.TestCase):
    def test_duplicate_process_identity_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(controller,'B',Path(d)):
            common.write(Path(d)/'CONTROLLER_IDENTITY.json',common.process_identity(os.getpid()))
            with self.assertRaises(AssertionError):controller.ensure_inactive()
    def test_release_requires_successful_audit_and_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);(b/'first_update.pt').write_bytes(b'synthetic checkpoint')
            common.write(b/'first_audit/CERTIFICATE.json',dict(state='pass',first_checkpoint_sha256=common.sha(b/'first_update.pt')))
            t=dict(exit_code=1,resource_stop_reason=None,cgroup_OOM_kill_delta=0)
            common.write(b/'first_audit/TERMINATION.json',t)
            with self.assertRaises(AssertionError):controller.release_first(b)
            t['exit_code']=0;common.write(b/'first_audit/TERMINATION.json',t);controller.release_first(b)
            self.assertTrue(controller.valid_release(b))
            (b/'first_update.pt').write_bytes(b'tampered')
            self.assertFalse(controller.valid_release(b))
    def test_each_budget_cap_is_checked(self):
        p=dict(per_fold_wall_seconds=100,per_fold_CPU_seconds=200,total5fold_wall_upper_seconds=400,total5fold_CPU_upper_seconds=800,combined_RSS_stop_KiB=1000)
        rows={str(i):dict(wall=0,cpu=0) for i in range(2,7)}
        self.assertIsNone(controller.budget_reason(rows,2,p,999))
        rows['2']['cpu']=201;self.assertEqual(controller.budget_reason(rows,2,p,0),'fold_CPU_budget')
        rows['2']=dict(wall=101,cpu=0);self.assertEqual(controller.budget_reason(rows,2,p,0),'fold_wall_budget')
        rows['2']=dict(wall=0,cpu=0);rows['3']['wall']=401;self.assertEqual(controller.budget_reason(rows,2,p,0),'total_wall_budget')
        rows['3']=dict(wall=0,cpu=801);self.assertEqual(controller.budget_reason(rows,2,p,0),'total_CPU_budget')
        rows['3']['cpu']=0;self.assertEqual(controller.budget_reason(rows,2,p,1001),'combined_RSS_budget')
if __name__=='__main__':unittest.main(verbosity=2)
