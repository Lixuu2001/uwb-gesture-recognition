"""Synthetic completion guards; no production arrays or checkpoints."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_audit import fixture,put
import audit_pretrained_phase as p
class Guards(unittest.TestCase):
 def test_missing_phase_before_arrays(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp)
   with patch.object(np,'load',side_effect=AssertionError('array')),patch.object(torch,'load',side_effect=AssertionError('checkpoint')):
    with self.assertRaisesRegex(RuntimeError,'not_ready'): p.audit(b,b/'out.json')
   self.assertFalse((b/'out.json').exists())
 def test_incomplete_arm_before_arrays(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);fixture(b);put(b/'run01/STATE.json',dict(completed_arms=[],users7_8_labels_used=False))
   with patch.object(np,'load',side_effect=AssertionError('array')),patch.object(torch,'load',side_effect=AssertionError('checkpoint')):
    with self.assertRaisesRegex(RuntimeError,'arm incomplete'): p.audit(b,b/'out.json')
 def test_bad_checkpoint_prediction_or_history_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);fixture(b);p.require_ready(b)
   for filename,key,value in [('run01/pretrained/ARM_COMPLETE.json','prediction_sha256','wrong'),('run01/pretrained/ARM_COMPLETE.json','classification_epochs',99),('run01/pretrained/classification/PHASE_COMPLETE.json','selected_epoch',99),('run01/STATE.json','users7_8_labels_used',True)]:
    path=b/filename;old=p.read(path);bad=copy.deepcopy(old);bad[key]=value;put(path,bad)
    with self.assertRaises(AssertionError):p.require_ready(b)
    put(path,old)
if __name__=='__main__':unittest.main(verbosity=2)
