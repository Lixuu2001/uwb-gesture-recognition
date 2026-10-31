"""Meaningful synthetic tests for this concrete pipeline, no real arrays or fit."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_pilot as runner
import audit_complete as auditor
from ops import *
class Tests(unittest.TestCase):
 def test_missing_marker_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'PROTOCOL.json').write_text(json.dumps(dict(seed=20261029,epochs=100,batch_size=16,cost_gate_passed=True,selection='source_final_last')))
   with patch.object(runner,'B',b),patch.object(runner,'ready',return_value=True):
    with self.assertRaises(FileNotFoundError):runner.main()
   self.assertFalse((b/'run01').exists())
 def test_bad_marker_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'PROTOCOL.json').write_text(json.dumps(dict(seed=20261029,epochs=100,batch_size=16,cost_gate_passed=True,selection='source_final_last')));(b/'REMOTE_PREREGISTRATION.json').write_text(json.dumps(dict(state='remote_preregistration_exact_readback_verified',protocol_sha256='wrong')))
   with patch.object(runner,'B',b),patch.object(runner,'ready',return_value=True):
    with self.assertRaises(AssertionError):runner.main()
   self.assertFalse((b/'run01').exists())
 def test_active_or_incomplete_rejects_audit(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'run01').mkdir()
   for state,pid in [('training',123),('incomplete',None),('complete',123)]:
    (b/'run01/STATE.json').write_text(json.dumps(dict(state=state,pid=pid)))
    with self.assertRaises(RuntimeError):auditor.require_complete(b)
 def test_roles_and_gate(self):
  ix=runner.roles(np.array([1,2,3,4,5,6]));self.assertEqual(ix['train'].tolist(),[2,3,4,5]);self.assertEqual(ix['validation'].tolist(),[1]);self.assertEqual(ix['diagnostic'].tolist(),[0]);self.assertFalse(runner.gate(.9866666666666667,.9866666666666667));self.assertTrue(runner.gate(.9916666666666667,.9866666666666667))
 def test_native_loader_checkpoint_eval_and_final_prediction_replay(self):
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);g=np.random.RandomState(31);x=g.randn(35,567,90).astype('float32');y=(np.arange(35)%12).astype('int64');ids=np.arange(35,dtype='int64');vx=g.randn(19,567,90).astype('float32');vy=(np.arange(19)%12).astype('int64');vi=np.arange(19,dtype='int64')+100;identity=dict(synthetic=True)
  def sequence(d):
   e=make(29);tr,vl=loaders(x,y,ids,vx,vy,vi);total=0;states={};rows=[];evalrows=[];epochs=[]
   states['initial']=snapshot(e,identity,0,0,-1,'before_loader_iteration')
   for ep in [1,2]:
    def observed(batch,sample,loss):
     nonlocal total
     total+=1;rows.append((ep,batch,sample[2].tolist(),loss))
     if total==1:
      p=Path(d)/'first.pt';runner.save(p,snapshot(e,identity,1,1,0,'after_optimizer_before_scheduler'));states['first']=torch.load(p,weights_only=False)
    stats=e.train_epoch(tr,observed);stats.update(monitor(e,tr,vl,ep,evalrows.append));epochs.append(stats)
   states['last']=snapshot(e,identity,2,total,2,'after_epoch_and_periodic_monitor');actual,logits=predictions(e,vx,vi);states['prediction']=snapshot(e,identity,2,total,2,'both_predictions_frozen_before_diagnostic_labels');return states,rows,evalrows,epochs,actual,logits
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   expected=sequence(a);actual=sequence(b);equal(expected,actual);self.assertEqual([len(r[2]) for r in actual[1]], [16,16,3]*2)
   first=expected[0]['first'];e=make(29);e.restore(first,expected_identity=identity);equal(first,snapshot(e,identity,1,1,0,'after_optimizer_before_scheduler'))
   fresh=make(29);dummy=torch.utils.data.DataLoader(torch.arange(35),batch_size=16,shuffle=True);self.assertEqual(next(iter(dummy)).tolist(),expected[1][0][2])
if __name__=='__main__':unittest.main()
