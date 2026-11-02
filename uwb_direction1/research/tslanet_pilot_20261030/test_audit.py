"""Synthetic/temporary fixtures only; never load production arrays/checkpoints."""
import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import audit_complete as a
torch.set_num_threads(1)
def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def fixture(base):
 run=base/'run01'
 put(run/'STATE.json',dict(state='complete',pid=None,completed_arms=['pretrained','scratch'],diagnostic_labels_read=True,users7_8_labels_used=False))
 hashes={}
 for arm,kind,n in a.PHASES:
  p=run/arm/kind;p.mkdir(parents=True,exist_ok=True)
  for key in ('initial','first_update','best','last'): (p/(key+'.pt')).write_bytes(b'not a real checkpoint')
  rows=[dict(arm=arm,phase=kind,epoch=i,training_records_used=4688,training_loss=1.,validation_loss=1/i,validation_macro=None if kind=='pretrain' else .9) for i in range(1,n+1)]
  (p/'epochs.jsonl').write_text('\n'.join(map(json.dumps,rows)))
  put(p/'PHASE_COMPLETE.json',dict(arm=arm,phase=kind,epochs=n,selected_epoch=n,checkpoint_sha256=a.digest(p/'best.pt')))
 for arm in ('pretrained','scratch'):
  (run/arm/'predictions_unscored.npz').write_bytes(b'not a real array');hashes[arm]=a.digest(run/arm/'predictions_unscored.npz')
  put(run/arm/'ARM_COMPLETE.json',dict(arm=arm,classification_epochs=100,pretraining_epochs=50 if arm=='pretrained' else 0,prediction_sha256=hashes[arm],checkpoint_sha256=a.digest(run/arm/'classification/best.pt')))
 put(run/'ALL_PREDICTIONS_FROZEN.json',hashes);put(run/'FILES.json',{});put(run/'SUMMARY.json',{})
 put(base/'supervisor01/TERMINATION.json',dict(exit_code=0,wall_timeout=False,supervisor_wall_seconds=100.,child_cpu_seconds=100.))

class AuditTests(unittest.TestCase):
 def test_full_guard_rejects_partial_before_any_array_or_checkpoint(self):
  with tempfile.TemporaryDirectory() as temp:
   b=Path(temp);fixture(b);put(b/'run01/STATE.json',dict(state='training',pid=123))
   with patch.object(np,'load',side_effect=AssertionError('array read')),patch.object(torch,'load',side_effect=AssertionError('checkpoint read')):
    with self.assertRaisesRegex(RuntimeError,'not_ready'): a.audit(b,b/'out.json')
   self.assertFalse((b/'out.json').exists())
 def test_full_guard_mutations(self):
  with tempfile.TemporaryDirectory() as temp:
   b=Path(temp);fixture(b);a.require_complete(b)
   for filename,mutation in [
    ('run01/STATE.json',lambda x:x.update(completed_arms=['pretrained'])),
    ('run01/STATE.json',lambda x:x.update(diagnostic_labels_read=False)),
    ('run01/STATE.json',lambda x:x.update(users7_8_labels_used=True)),
    ('supervisor01/TERMINATION.json',lambda x:x.update(exit_code=1)),
    ('supervisor01/TERMINATION.json',lambda x:x.update(wall_timeout=True)),
    ('supervisor01/TERMINATION.json',lambda x:x.update(child_cpu_seconds=43201)),
    ('run01/ALL_PREDICTIONS_FROZEN.json',lambda x:x.update(pretrained='wrong')),
    ('run01/pretrained/pretrain/PHASE_COMPLETE.json',lambda x:x.update(epochs=49)),
    ('run01/scratch/classification/PHASE_COMPLETE.json',lambda x:x.update(selected_epoch=99))]:
    p=b/filename;old=a.read(p);bad=copy.deepcopy(old);mutation(bad);put(p,bad)
    with self.assertRaises((AssertionError,RuntimeError)): a.require_complete(b)
    put(p,old)
   (b/'run01/ALL_PREDICTIONS_FROZEN.json').unlink()
   with self.assertRaises(FileNotFoundError): a.require_complete(b)
 def test_ssl_guard_before_any_array_or_checkpoint(self):
  with tempfile.TemporaryDirectory() as temp:
   b=Path(temp)
   with patch.object(np,'load',side_effect=AssertionError('array read')),patch.object(torch,'load',side_effect=AssertionError('checkpoint read')):
    with self.assertRaisesRegex(RuntimeError,'phase incomplete'): a.audit_ssl_phase(b,b/'out.json')
   self.assertFalse((b/'out.json').exists())
 def test_history_and_gate_priority(self):
  rows=[dict(epoch=3,validation_loss=.2,validation_macro=.9),dict(epoch=1,validation_loss=.1,validation_macro=.8),dict(epoch=2,validation_loss=.2,validation_macro=.9)]
  self.assertEqual(a.choose(rows,'pretrain')['epoch'],1);self.assertEqual(a.choose(rows,'classification')['epoch'],2)
  cases=[(.99,.98,.9866666666666667,'pretrained_mechanism'),(.99,.995,.9866666666666667,'scratch_baseline'),(.995,.991,.9866666666666667,'pretrained_baseline'),(.98,.98,.9866666666666667,'close_exact_configuration')]
  for p,s,r,route in cases: self.assertEqual(a.decision(dict(pretrained=p,scratch=s),r)['predeclared_route'],route)
  self.assertTrue(a.decision(dict(pretrained=.99,scratch=.985),.995)['mechanism_expansion_eligible'])
  self.assertAlmostEqual(a.metric(np.array([0,0,2]),np.array([0,1,2])),.75)
 def test_roles_reject_overlap_and_non_source_users(self):
  users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);ids=np.arange(6900)
  split=dict(train=ids[np.isin(users,[3,4,5,6])].tolist(),validation=ids[users==2].tolist(),diagnostic=ids[users==1].tolist())
  ix=a.role_indices(split,ids,users);self.assertEqual(len(ix['train']),4700)
  split['train'][0]=split['diagnostic'][0]
  with self.assertRaises(AssertionError): a.role_indices(split,ids,users)
  users[-1]=7
  with self.assertRaises(AssertionError): a.role_indices(split,ids,users)
 def test_adamw_shape_steps_unused_and_reload(self):
  model=torch.nn.Linear(3,2);model.register_parameter('unused',torch.nn.Parameter(torch.ones(4)))
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  model(torch.ones(2,3)).sum().backward();optimizer.step();saved=optimizer.state_dict()
  self.assertEqual(a.optimizer_check(model,saved,1,{'unused'}),2)
  for key,value in [('lr',.01),('weight_decay',.001),('betas',(.8,.999))]:
   bad=copy.deepcopy(saved);bad['param_groups'][0][key]=value
   with self.assertRaises(AssertionError): a.optimizer_check(model,bad,1,{'unused'})
  with self.assertRaises(AssertionError): a.optimizer_check(model,saved,2,{'unused'})
 def test_independent_smoothing_and_preprocessing(self):
  torch.manual_seed(9);logits=torch.randn(16,12);y=torch.arange(16)%12;lp=torch.log_softmax(logits,-1)
  loss=(.9*(-lp.gather(1,y[:,None]).squeeze(1))+.1*(-lp.mean(-1))).mean()
  torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(logits,y,label_smoothing=.1))
  raw=np.arange(2*3*90*189,dtype=np.float32).reshape(2,3,90,189)/10000
  got=a.preprocess(raw);self.assertEqual(got.shape,(2,90,567))
  for i in range(2):
   for v in range(3):
    expected=raw[i,v]/max(float(np.linalg.norm(raw[i,v]))/np.sqrt(90*189),1e-8)
    np.testing.assert_allclose(got[i,:,v*189:(v+1)*189],expected,rtol=2e-6,atol=2e-6)
if __name__=='__main__': unittest.main(verbosity=2)
