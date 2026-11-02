"""Separate low-cost synthetic control-plane guard;no model/data/fit imports."""
from pathlib import Path
import tempfile,unittest,json
from unittest.mock import patch
import controller
class Policy(unittest.TestCase):
    def test_recorded_RSS_stop_blocks_audit_even_if_worker_exit_zero(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            result=dict(exit_code=0,wall_timeout=False,resource_stop_reason='RSS_resource_limit',scientific_negative=False)
            with patch.object(controller,'B',root),patch.object(controller,'ready',return_value={}),patch.object(controller,'bounded_run',return_value=result) as run:
                controller.main()
            state=json.loads((root/'STATE.json').read_bytes())
            self.assertEqual(run.call_count,1);self.assertEqual(state['state'],'execution_incomplete_producer')
            self.assertFalse(state['scientific_negative']);self.assertFalse(state['diagnostic_scored'])
if __name__=='__main__':unittest.main(verbosity=2)
