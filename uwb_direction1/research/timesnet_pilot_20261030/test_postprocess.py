import json,tempfile,unittest
from pathlib import Path
from postprocess import training_ready
class Gate(unittest.TestCase):
 def test_wait_no_termination(self):
  with tempfile.TemporaryDirectory() as d:self.assertFalse(training_ready(Path(d)))
 def test_reject_failed_or_mismatched_training(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'supervisor01').mkdir();(p/'run01').mkdir();f=p/'supervisor01/TERMINATION.json'
   for t in [{'exit_code':1,'wall_timeout':False},{'exit_code':0,'wall_timeout':True}]:
    f.write_text(json.dumps(t))
    with self.assertRaisesRegex(RuntimeError,'training_failed'):training_ready(p)
   f.write_text(json.dumps({'exit_code':0,'wall_timeout':False}));(p/'run01/STATE.json').write_text(json.dumps({'state':'training','pid':123}))
   with self.assertRaisesRegex(RuntimeError,'mismatch'):training_ready(p)
 def test_accept_success_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'supervisor01').mkdir();(p/'run01').mkdir();(p/'supervisor01/TERMINATION.json').write_text(json.dumps({'exit_code':0,'wall_timeout':False}));(p/'run01/STATE.json').write_text(json.dumps({'state':'complete','pid':None}));self.assertTrue(training_ready(p))
if __name__=='__main__':unittest.main(verbosity=2)
