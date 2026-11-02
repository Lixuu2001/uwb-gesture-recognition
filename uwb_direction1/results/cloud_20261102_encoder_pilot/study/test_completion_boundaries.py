"""Refuse incomplete runs and partial insurance before any score/data load."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import guards,finalize
class Tests(unittest.TestCase):
 def test_active_training_refused(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'run01').mkdir();(b/'supervisor01').mkdir()
   (b/'run01/STATE.json').write_text(json.dumps({'state':'training','pid':123,'all_predictions_frozen':False,'diagnostic_labels_read':False,'users7_8_labels_used':False}))
   (b/'supervisor01/TERMINATION.json').write_text(json.dumps({'exit_code':0,'wall_timeout':False}))
   with self.assertRaises(RuntimeError):guards.require_training_complete(b)
 def test_partial_private_scope_cannot_score(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);p=b/'complete_backup';p.mkdir()
   for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json']:(p/n).write_text(json.dumps({'scope':'initial_first_only_not_latest_or_complete','complete_modelset_verified':False}))
   with patch.object(finalize,'B',b),patch.object(finalize,'complete',return_value={}):
    with self.assertRaises(AssertionError):finalize.main()
 def test_failed_training_refused(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'run01').mkdir();(b/'supervisor01').mkdir()
   (b/'run01/STATE.json').write_text(json.dumps({'state':'complete','pid':None,'all_predictions_frozen':True,'diagnostic_labels_read':False,'users7_8_labels_used':False}))
   (b/'supervisor01/TERMINATION.json').write_text(json.dumps({'exit_code':-9,'wall_timeout':True}))
   with self.assertRaises(RuntimeError):guards.require_training_complete(b)
if __name__=='__main__':unittest.main()
