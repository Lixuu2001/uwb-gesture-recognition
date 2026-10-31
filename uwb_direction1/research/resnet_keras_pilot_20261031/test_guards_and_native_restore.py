"""Synthetic only: preregistration/active-run guards and full native checkpoint replay."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_pilot as runner
import audit_complete as auditor
from ops import *
class Guards(unittest.TestCase):
 def test_missing_remote_marker_prevents_output_creation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'PROTOCOL.json').write_text(json.dumps(dict(seed=20261029,epochs=100,batch_size=64,parameters=834124,cost_gate_passed=True)))
   with patch.object(runner,'B',root),patch.object(runner,'ready',return_value=True):
    with self.assertRaises(FileNotFoundError):runner.main()
   self.assertFalse((root/'run01').exists())
 def test_wrong_remote_protocol_hash_prevents_output_creation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'PROTOCOL.json').write_text(json.dumps(dict(seed=20261029,epochs=100,batch_size=64,parameters=834124,cost_gate_passed=True)));(root/'REMOTE_PREREGISTRATION.json').write_text(json.dumps(dict(state='remote_preregistration_exact_readback_verified',protocol_sha256='wrong')))
   with patch.object(runner,'B',root),patch.object(runner,'ready',return_value=True):
    with self.assertRaises(AssertionError):runner.main()
   self.assertFalse((root/'run01').exists())
 def test_active_or_incomplete_cannot_audit(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'run01').mkdir()
   for state,pid in [('training',123),('incomplete',None),('complete',123)]:
    (root/'run01/STATE.json').write_text(json.dumps(dict(state=state,pid=pid)))
    with self.assertRaises(RuntimeError):auditor.require_complete(root)
 def test_bad_previous_state_prevents_progress(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.json';p.write_text(json.dumps(dict(state='training',pid=123)))
   with self.assertRaises(AssertionError):runner.ready(dict(previous_state=str(p)))
 def test_fixed_partitions_and_gate(self):
  for ep in [1,100]:self.assertEqual(sorted(training_order(ep).tolist()),list(range(4700)))
  self.assertEqual([len(training_order(1)[i:i+64]) for i in range(0,4700,64)], [64]*73+[28]);self.assertEqual([len(evaluation_order(1000,'validation')[i:i+64]) for i in range(0,1000,64)], [64]*15+[40]);self.assertEqual([len(evaluation_order(1200,'diagnostic')[i:i+64]) for i in range(0,1200,64)], [64]*18+[48]);self.assertFalse(runner.gate(.9866666666666667,.9866666666666667));self.assertTrue(runner.gate(.9916666666666667,.9866666666666667))
 def test_native_checkpoint_and_callback_restore(self):
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);g=np.random.RandomState(5);x=g.randn(70,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(70)%12];vx=g.randn(17,90,567).astype('float32');vy=np.arange(17)%12;identity=dict(synthetic_only=True)
  def sequence(d,restore_at=None):
   h,m,cb=make(d);total=0;states={};epochs=[];batches=[]
   states['initial']=copy.deepcopy(snapshot(h,m,0,0,-1,identity))
   for ep in range(1,4):
    cb.on_epoch_begin(ep-1);m.reset_metrics()
    for j,i in enumerate(range(0,70,64)):
     cb.on_train_batch_begin(j);logs,loss=train_step(m,x[i:i+64],y[i:i+64]);cb.on_train_batch_end(j,logs);total+=1;batches.append((logs,loss))
     if total==1:states['first']=copy.deepcopy(snapshot(h,m,ep,total,j,identity))
    logs=dict(logs);logs.update({'val_'+k:v for k,v in validation(m,vx,vy).items()});cb.on_epoch_end(ep-1,logs);epochs.append(logs);states['last']=copy.deepcopy(snapshot(h,m,ep,total,j,identity))
    if restore_at==ep:
     saved=copy.deepcopy(states['last']);runner.save(Path(d)/'resume.pt',saved);saved=torch.load(Path(d)/'resume.pt',weights_only=False);h,m,cb=make(d);restore(saved,h,m);equal(saved,snapshot(h,m,ep,total,j,identity))
   cb.on_train_end();return states,epochs,batches
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as c:
   gold=sequence(a);replay=sequence(b);resumed=sequence(c,restore_at=1);equal(gold,replay);equal(gold,resumed)
   bad=copy.deepcopy(gold[0]['first']);bad['native']['optimizer'][0]+=1
   with self.assertRaises(AssertionError):equal(bad,gold[0]['first'])
if __name__=='__main__':unittest.main()
