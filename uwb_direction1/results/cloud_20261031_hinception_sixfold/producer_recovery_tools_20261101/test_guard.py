"""Only synthetic state metadata, no UWB values/labels or fitting."""
from pathlib import Path
from unittest.mock import patch
import json,tempfile,unittest
import producer_snapshot as m
class Tests(unittest.TestCase):
 def test_active_refuses_before_termination_or_model_hash(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);p=b/'run01/held5';p.mkdir(parents=True);(p/'STATE.json').write_text(json.dumps(dict(state='training',pid=123,all_predictions_frozen=False,diagnostic_labels_read=False,users7_8_labels_used=False)))
   with patch.object(m,'sha',side_effect=AssertionError('Must not hash a model')) as h:
    with self.assertRaisesRegex(RuntimeError,'still active/incomplete'):m.validate(b,{},5)
    h.assert_not_called()
 def test_other_producer_refuses_before_any_file(self):
  with patch.object(m,'read',side_effect=AssertionError('No prerequisite file read')) as r:
   with self.assertRaisesRegex(RuntimeError,'Only remaining'):m.validate(Path('/absent'),{},2)
   r.assert_not_called()
if __name__=='__main__':unittest.main()
