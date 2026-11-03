"""UNEXECUTED while executor unavailable; synthetic route boundaries only."""
import unittest
from recovery_probe import choose_route,successful_term,budget_ok
def good_term():
    return dict(exit_code=0,wall_timeout=False,resource_stop_reason=None,
                cgroup_oom_kill_delta=0,supervisor_wall_seconds=100,
                child_cpu_seconds=350,peak_observed_RSS_KiB=2000000)
def state():
    return dict(summary_exists=False,root_exists=True,
        frozen_controls_verified=True,producer_term=good_term(),
        audit_term=good_term(),producer_live=False,audit_live=False,
        certificate_verified=True,budget_verified=True,
        dependencies_verified=True,private_directory_exists=False)
class Boundaries(unittest.TestCase):
    def test_exit_zero_with_resource_stop_does_not_pass(self):
        t=good_term();t['resource_stop_reason']='RSS_resource_limit'
        self.assertFalse(successful_term(t))
        t=good_term();t.pop('cgroup_oom_kill_delta')
        self.assertFalse(successful_term(t))
    def test_unknown_dead_process_is_not_automatic_resume(self):
        s=state();s.update(producer_term=None,producer_live=False,audit_term=None)
        self.assertEqual(choose_route(s),'producer_not_verified_inspect_evidence_no_auto_resume')
    def test_existing_summary_never_rescores_even_if_runtime_missing(self):
        s=state();s.update(summary_exists=True,root_exists=False)
        self.assertEqual(choose_route(s),'already_finalized_inspect_existing_provenance_do_not_rescore')
    def test_partial_certificate_or_unverified_dependencies_blocks(self):
        s=state();s['certificate_verified']=False
        self.assertEqual(choose_route(s),'blocked_missing_or_incomplete_full_audit_certificate')
        s=state();s['dependencies_verified']=False
        self.assertEqual(choose_route(s),'verify_dependencies_before_original_complete_guard')
    def test_existing_private_directory_requires_reuse_and_original_verification(self):
        s=state();s['private_directory_exists']=True
        self.assertEqual(choose_route(s),'inspect_existing_private_backup_reuse_known_IDs_original_stream_verify_required')
    def test_budget_and_missing_workspace_are_not_accuracy_results(self):
        p,a=good_term(),good_term();a['child_cpu_seconds']=172801
        self.assertFalse(budget_ok(p,a))
        s=state();s['root_exists']=False
        self.assertEqual(choose_route(s),'workspace_identity_unverified_inspect_persistent_index_not_data_loss')
if __name__=='__main__':unittest.main(verbosity=2)
