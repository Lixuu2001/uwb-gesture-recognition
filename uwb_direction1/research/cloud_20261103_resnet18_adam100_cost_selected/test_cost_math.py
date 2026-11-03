"""Synthetic arithmetic boundary fixtures, not hardware performance results."""
import json,unittest
from pathlib import Path
from cost_math import estimate
P=json.loads(Path(__file__).with_name('COST_POLICY_DRAFT.json').read_text())
def result(step=.5,rss=1000000,free=1000000000):
    return estimate(P,[step]*6,[.1]*3,[.02]*3,[.02]*3,[.01]*3,[.01]*3,
                    134200000,44750000,rss,free)
class Boundaries(unittest.TestCase):
    def test_complete_100epochs_plus_replay_cannot_fit_one_second_updates(self):
        self.assertTrue(result(.5)['resource_gate_passed'])
        self.assertFalse(result(1.)['resource_gate_passed'])
        self.assertEqual(result()['full_training_updates'],23500)
        self.assertEqual(result()['per_run_validation_batches'],5000)
    def test_full_checkpoint_and_streaming_workspace_boundary(self):
        disk=result()['required_additional_workspace_bytes']
        self.assertTrue(result(free=disk)['resource_gate_passed'])
        self.assertFalse(result(free=disk-1)['resource_gate_passed'])
        self.assertGreater(disk,500*1048576)
    def test_no_resource_claim_for_nonfinite_or_incomplete_timings(self):
        with self.assertRaises(AssertionError):result(float('nan'))
        with self.assertRaises(AssertionError):
            estimate(P,[.5]*5,[.1]*3,[.02]*3,[.02]*3,[.01]*3,[.01]*3,
                     134200000,44750000,1000000,1000000000)
    def test_memory_margin_and_no_epoch_shrinking(self):
        self.assertFalse(result(rss=13*1048576)['resource_gate_passed'])
        q=dict(P);q['epochs']=40
        with self.assertRaises(AssertionError):
            estimate(q,[.5]*6,[.1]*3,[.02]*3,[.02]*3,[.01]*3,[.01]*3,
                     134200000,44750000,1000000,1000000000)
if __name__=='__main__':unittest.main(verbosity=2)
