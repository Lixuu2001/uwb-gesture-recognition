"""Temporary guards only; no UWB arrays/models or fitting."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import audit_complete as a
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def fixture(b):
 out=b/'run01';out.mkdir();put(out/'STATE.json',dict(state='complete',pid=None,iteration=600,all_predictions_frozen=True,diagnostic_labels_read=True,users7_8_labels_used=False))
 for n in ('last.pt','initial.pt','first_update.pt','features.npz','svm.joblib','ITERATIONS.jsonl','SVM_TRAINING_CV_IDS.json'):(out/n).write_bytes(b'fake')
 for n in ('FILES.json','SUMMARY.json'):put(out/n,{})
 frozen={}
 for role in ('validation','diagnostic'):
  p=out/(role+'_predictions_unscored.npz');p.write_bytes(b'not an array');frozen[role]=a.sha(p)
 put(out/'ALL_PREDICTIONS_FROZEN.json',frozen);put(out/'ENCODER_COMPLETE.json',dict(iterations=600,initial_inclusive_average_samples=601,raw_and_average_checkpoint_sha256=a.sha(out/'last.pt')));put(b/'supervisor01/TERMINATION.json',dict(exit_code=0,wall_timeout=False,supervisor_wall_seconds=1,child_cpu_seconds=1))
class Guards(unittest.TestCase):
 def test_incomplete_before_any_array_model(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);fixture(b);put(b/'run01/STATE.json',dict(state='training',pid=5,iteration=599))
   with patch.object(np,'load',side_effect=AssertionError('array')),patch.object(torch,'load',side_effect=AssertionError('model')):
    with self.assertRaisesRegex(RuntimeError,'not_ready'):a.audit(b,b/'out.json')
   self.assertFalse((b/'out.json').exists())
 def test_supervisor_freeze_and_required_artifacts(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);fixture(b);a.require_complete(b);term=b/'supervisor01/TERMINATION.json';old=a.read(term);bad=dict(old,exit_code=1);put(term,bad)
   with self.assertRaises(RuntimeError):a.require_complete(b)
   put(term,old);put(b/'run01/ALL_PREDICTIONS_FROZEN.json',dict(validation='wrong',diagnostic='wrong'))
   with self.assertRaises(AssertionError):a.require_complete(b)
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);fixture(b);(b/'run01/svm.joblib').unlink()
   with self.assertRaisesRegex(RuntimeError,'missing'):a.require_complete(b)
if __name__=='__main__':unittest.main(verbosity=2)
