"""Synthetic tamper/role/numeric/optimizer checks; no real arrays or outcomes."""
import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import audit_complete as audit
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
from adapter import StableLookahead

class AuditTests(unittest.TestCase):
 def fixture(self,root):
  run=root/'run01';run.mkdir();(root/'supervisor01').mkdir()
  def write(p,v):p.write_text(json.dumps(v))
  write(run/'STATE.json',dict(state='complete',pid=None,completed_folds=list(range(1,7)),all_new_diagnostic_predictions_frozen=True))
  write(root/'supervisor01/TERMINATION.json',dict(exit_code=0,wall_timeout=False,supervisor_wall_seconds=100,child_cpu_seconds=100))
  frozen={}
  for h in range(2,7):
   d=run/f'held{h}';d.mkdir();(d/'best.pt').write_bytes(b'synthetic checkpoint');(d/'predictions_unscored.npz').write_bytes(b'synthetic predictions')
   frozen[str(h)]=audit.digest(d/'predictions_unscored.npz')
   write(d/'FOLD_COMPLETE.json',dict(held=h,epochs=64,prediction_sha256=frozen[str(h)],checkpoint_sha256=audit.digest(d/'best.pt')))
  write(run/'ALL_NEW_PREDICTIONS_FROZEN.json',frozen)
  return run
 def test_ready_metadata_only_fixture(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.fixture(p)
   with patch('numpy.load',side_effect=AssertionError('arrays must remain closed')):self.assertEqual(audit.require_complete(p)['exit_code'],0)
 def test_active_run_refuses_entire_audit_before_array_loading(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);r=self.fixture(p);state=json.loads((r/'STATE.json').read_text());state.update(state='training',pid=123);(r/'STATE.json').write_text(json.dumps(state))
   with patch('numpy.load') as loader:
    with self.assertRaisesRegex(RuntimeError,'not_ready'):audit.audit(p,p/'not_created.json')
    loader.assert_not_called()
   self.assertFalse((p/'not_created.json').exists())
 def test_freeze_and_fold_guards(self):
  for change in ('missing_fold','not_frozen','old_epoch'):
   with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
    p=Path(d);r=self.fixture(p)
    if change=='old_epoch':
     q=r/'held4/FOLD_COMPLETE.json';v=json.loads(q.read_text());v['epochs']=63;q.write_text(json.dumps(v))
    else:
     q=r/'STATE.json';v=json.loads(q.read_text())
     if change=='missing_fold':v['completed_folds']=[1,2,3,4,5]
     else:v['all_new_diagnostic_predictions_frozen']=False
     q.write_text(json.dumps(v))
    with self.assertRaises((RuntimeError,AssertionError)):audit.require_complete(p)
 def test_budget_failure_guards(self):
  for field,value in [('exit_code',1),('wall_timeout',True),('supervisor_wall_seconds',43201),('child_cpu_seconds',43201)]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    p=Path(d);self.fixture(p);q=p/'supervisor01/TERMINATION.json';v=json.loads(q.read_text());v[field]=value;q.write_text(json.dumps(v))
    with self.assertRaises((RuntimeError,AssertionError)):audit.require_complete(p)
 def test_tampered_prediction_and_checkpoint_rejected(self):
  for name in ('best.pt','predictions_unscored.npz'):
   with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
    p=Path(d);r=self.fixture(p);(r/'held3'/name).write_bytes(b'changed')
    with self.assertRaises(AssertionError):audit.require_complete(p)
 def test_source_roles_disjoint_and_tamper(self):
  users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);ids=np.arange(6900)*2+17
  for held in range(1,7):
   val=held%6+1;split=dict(train=ids[(users!=held)&(users!=val)],validation=ids[users==val],diagnostic=ids[users==held])
   ix=audit.split_indices(split,ids,users,held);self.assertEqual(set(users[ix['train']]),set(range(1,7))-{held,val})
   bad={k:v.copy() for k,v in split.items()};bad['train'][0]=bad['diagnostic'][0]
   with self.assertRaises(AssertionError):audit.split_indices(bad,ids,users,held)
 def test_independent_preprocess_signed_and_zero_maps(self):
  torch.set_num_threads(1);x=np.random.default_rng(5).normal(size=(4,3,90,189)).astype('float32');x[0,1]=0
  tx=torch.from_numpy(x);expected=(tx/tx.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)).permute(0,2,1,3).reshape(4,90,567).numpy()
  value=audit.preprocess(x);np.testing.assert_allclose(value,expected,rtol=2e-5,atol=2e-6);self.assertTrue(np.isfinite(value).all());self.assertEqual(value.shape,(4,90,567))
 def test_selection_macro_loss_epoch_and_old_history_schema(self):
  rows=[dict(epoch=1,validation_macro=.8,validation_bce=.1),dict(epoch=2,validation_macro=.9,validation_bce=.2),dict(epoch=3,validation_macro=.9,validation_bce=.1),dict(epoch=4,validation_macro=.9,validation_bce=.1)]
  normalized=audit.normalized_history(rows,True);self.assertEqual(audit.choose(normalized)['epoch'],3);self.assertNotIn('validation_loss',rows[0]);self.assertIs(audit.normalized_history(normalized,False),normalized)
 def test_macro_and_gate_independent_values(self):
  self.assertEqual(audit.metric(np.array([0,0,4,4]),np.array([0,1,4,4])),.75)
  self.assertTrue(audit.gate([.02]*6));self.assertFalse(audit.gate([.009]*6));self.assertFalse(audit.gate([.05,.05,.05,-.001,-.001,-.001]));self.assertFalse(audit.gate([.04,.04,.04,.04,.04,-.021]))
 def test_optimizer_initial_and_slow_buffer_reload_tamper(self):
  torch.set_num_threads(1);model=torch.nn.Linear(3,2);opt=StableLookahead(model.parameters())
  for step in range(1,7):
   opt.zero_grad();model(torch.ones(2,3)).square().sum().backward();opt.step()
   if step in (1,6):audit.optimizer_check(model,opt.state_dict(),step,StableLookahead)
  bad=copy.deepcopy(opt.state_dict());bad['fast']['state'][0]['exp_avg'][0,0]=float('nan')
  with self.assertRaises(AssertionError):audit.optimizer_check(model,bad,6,StableLookahead)
  bad=copy.deepcopy(opt.state_dict());bad['slow'].pop(0)
  with self.assertRaises(AssertionError):audit.optimizer_check(model,bad,6,StableLookahead)
  bad=copy.deepcopy(opt.state_dict());bad['count']=5
  with self.assertRaises(AssertionError):audit.optimizer_check(model,bad,6,StableLookahead)

if __name__=='__main__':unittest.main()
