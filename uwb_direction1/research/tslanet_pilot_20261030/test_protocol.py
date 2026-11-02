"""Synthetic user-boundary and fixed gate-priority checks; no UWB data loads."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_pilot as run
class Tests(unittest.TestCase):
 def test_roles_and_all_source_counts(self):
  users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);i=run.roles(users)
  self.assertEqual({k:len(v) for k,v in i.items()},dict(train=4700,validation=1000,diagnostic=1200))
  self.assertEqual(set(users[i['train']]),{3,4,5,6});self.assertEqual(set(users[i['validation']]),{2});self.assertEqual(set(users[i['diagnostic']]),{1})
  self.assertFalse(set(i['train'])&set(i['validation']) or set(i['train'])&set(i['diagnostic']) or set(i['validation'])&set(i['diagnostic']))
 def test_gate_mechanism_then_scratch_then_pretrained_priority(self):
  self.assertEqual(run.gates(dict(pretrained=.96,scratch=.94),.95)['predeclared_route'],'pretrained_mechanism')
  self.assertEqual(run.gates(dict(pretrained=.97,scratch=.98),.95)['predeclared_route'],'scratch_baseline')
  self.assertEqual(run.gates(dict(pretrained=.956,scratch=.954),.95)['predeclared_route'],'pretrained_baseline')
  self.assertEqual(run.gates(dict(pretrained=.945,scratch=.947),.95)['predeclared_route'],'close_exact_configuration')
  self.assertFalse(run.gates(dict(pretrained=.94,scratch=.92),.95)['mechanism_expansion_eligible'])
  self.assertTrue(run.gates(dict(pretrained=.955,scratch=.95),.95)['mechanism_expansion_eligible'])
 def fixture(self,p):
  cfg={}
  for key,value in dict(previous_state=dict(state='complete',pid=None),previous_audit=dict(state='pass'),previous_private_backup=dict(state='private_readback_verified'),previous_termination=dict(exit_code=0,wall_timeout=False)).items():
   q=p/(key+'.json');q.write_text(json.dumps(value));cfg[key]=str(q)
  return cfg
 def test_prior_gate_never_opens_arrays(self):
  with tempfile.TemporaryDirectory() as d:
   cfg=self.fixture(Path(d))
   with patch('numpy.load',side_effect=AssertionError('array access forbidden')):run.ready(cfg)
   Path(cfg['previous_state']).write_text(json.dumps(dict(state='training',pid=10)))
   with self.assertRaisesRegex(RuntimeError,'not_ready'):run.ready(cfg)
 def test_incomplete_audit_backup_and_supervisor_rejected(self):
  for key,value in [('previous_audit',dict(state='incomplete')),('previous_private_backup',dict(state='upload_only')),('previous_termination',dict(exit_code=1,wall_timeout=False)),('previous_termination',dict(exit_code=0,wall_timeout=True))]:
   with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
    cfg=self.fixture(Path(d));Path(cfg[key]).write_text(json.dumps(value))
    with self.assertRaises(AssertionError):run.ready(cfg)
 def test_macro_missing_classes(self):self.assertEqual(run.macro(np.array([0,0,3,3]),np.array([0,1,3,3])),.75)
if __name__=='__main__':unittest.main()
