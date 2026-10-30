"""Temporary fake prerequisites and intercepted startup, never real UWB values."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import run_pilot as r
def write(path,obj):path.write_text(json.dumps(obj))
def fixture(b):
 cfg={}
 for key,value in [('previous_state',dict(state='complete',pid=None)),('previous_termination',dict(exit_code=0,wall_timeout=False)),('previous_audit',dict(state='pass')),('previous_summary',dict(gates=dict(predeclared_route='close_exact_configuration'),scores=dict(pretrained=.97,scratch=.96))),('previous_final_private_backup',dict(state='private_readback_verified',full_experiment_modelset_recoverable_composite=True)),('prior_held1_summary',dict(scores=dict(timemil=.98,gap=.975),cnn_reference=dict(full=.95,local=.96)))]:
  p=b/(key+'.json');write(p,value);cfg[key]=str(p)
 return cfg
class ProtocolTests(unittest.TestCase):
 def test_prior_ready_and_eligible_route(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);cfg=fixture(b);r.ready(cfg)
   for key,field,value in [('previous_state','state','training'),('previous_state','pid',123),('previous_termination','exit_code',1),('previous_termination','wall_timeout',True),('previous_audit','state','fail'),('previous_summary','gates',dict(predeclared_route='pretrained_mechanism')),('previous_final_private_backup','full_experiment_modelset_recoverable_composite',False)]:
    p=Path(cfg[key]);original=r.read(p);bad=copy.deepcopy(original);bad[field]=value;write(p,bad)
    with self.assertRaises((RuntimeError,AssertionError)):r.ready(cfg)
    write(p,original)
 def test_roles_and_bounded_gate(self):
  users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);ix=r.roles(users);self.assertEqual({k:len(v) for k,v in ix.items()},dict(train=4700,validation=1000,diagnostic=1200))
  self.assertEqual(set(users[ix['train']]),{3,4,5,6});self.assertEqual(set(users[ix['validation']]),{2});self.assertEqual(set(users[ix['diagnostic']]),{1})
  self.assertTrue(r.gate(.995,.99));self.assertFalse(r.gate(.994,.99));self.assertFalse(r.gate(.99,.99))
 def test_actual_startup_reaches_intercepted_array_boundary(self):
  class Boundary(Exception):pass
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);cfg=fixture(b);source=b/'fake_source.npz';source.write_bytes(b'not an array')
   cfg.update(iterations=600,seed=20261029,batch_size=16,cost_gate_passed=True,reference_held1_macro=.98,source_manifest=str(source),source_manifest_sha256=r.sha(source))
   write(b/'PROTOCOL.json',cfg);write(b/'FROZEN_HASHES.json',{'PROTOCOL.json':r.sha(b/'PROTOCOL.json')})
   with patch.object(r,'B',b),patch.object(r.np,'load',side_effect=Boundary('synthetic interception')),patch.object(r.torch,'set_num_threads'),patch.object(r.torch,'set_num_interop_threads'),patch.object(r.resource,'setrlimit'),patch.object(r.signal,'setitimer'),patch.object(r.signal,'signal'),patch.object(r,'make_model',side_effect=AssertionError('model reached')):
    with self.assertRaises(Boundary):r.main()
   state=r.read(b/'run01/STATE.json');self.assertEqual(state['state'],'incomplete');self.assertFalse(state['diagnostic_labels_read']);self.assertFalse(state['scientific_negative']);self.assertFalse((b/'run01/initial.pt').exists())
if __name__=='__main__':unittest.main(verbosity=2)
