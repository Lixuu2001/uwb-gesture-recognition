"""Policy boundaries only: do not run the network or inspect any UWB labels."""
import json,unittest
from pathlib import Path
from cost_math import select
P=json.loads((Path(__file__).parent/'POLICY.json').read_bytes())
class Boundaries(unittest.TestCase):
 def result(self,seconds=1,rss=1000000,free=10**10):return select(P,[seconds]*6,[seconds],[.1]*3,[.1]*3,[.1]*3,10351235,rss,free)
 def test_full_training_and_replay_budget(self):
  self.assertTrue(self.result(1)['resource_gate_passed']);self.assertFalse(self.result(2)['resource_gate_passed'])
 def test_memory_headroom(self):self.assertFalse(self.result(rss=13*1048576)['resource_gate_passed'])
 def test_complete_private_archive_storage_not_head_only(self):
  r=self.result();self.assertGreater(r['required_additional_workspace_bytes'],150*1048576);self.assertFalse(self.result(free=r['required_additional_workspace_bytes']-1)['resource_gate_passed'])
 def test_cannot_silently_shrink_training(self):
  q=dict(P);q['epochs']=20
  with self.assertRaises(AssertionError):select(q,[1]*6,[1],[.1]*3,[.1]*3,[.1]*3,10351235,1000000,10**10)
if __name__=='__main__':unittest.main(verbosity=2)
