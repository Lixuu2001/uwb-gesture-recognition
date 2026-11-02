"""Synthetic record fixtures only; none are actual timing/performance results."""
import copy,json,unittest,tempfile,hashlib,sys
from pathlib import Path
from unittest.mock import patch
from cost_evidence import *
class Evidence(unittest.TestCase):
 def fixture(self):
  p=dict(epoch_candidates=[3,1],batch_size=8,seed=7,input_preprocessing='synthetic_metadata_fixture_only',budget_seconds=43200,memory_limit_KiB=1024,prior_workspace='synthetic_prior_not_a_real_study')
  raw=json.dumps(p,indent=2).encode();vectors=dict(training_seconds=[.2]*8,evaluation_seconds=[.1]*4,keras_save_seconds=[.02]*3,snapshot_save_seconds=[.01]*3,max_RSS_KiB=500)
  r=select(p,**vectors);r.update(vectors);r.update(policy=p,full_author_default_DCNN2L_architecture=True,total_parameters=2587468,trainable_parameters=2586956,original_training_classweight_formula=True,validation_unweighted=True,synthetic_fixture_only=True,preprocessing_choice_recorded_but_no_UWB_preprocessing_executed=True,real_UWB_values_labels_or_fit=False,real_pilot_started=False)
  d=dict(policy_sha256=hashlib.sha256(raw).hexdigest(),utility_sha256='synthetic_source_identity',policy_fixed_before_timing=True,real_UWB_values_labels_or_fit=False,declared_unix=1)
  t=dict(exit_code=0,wall_timeout=False,supervisor_wall_seconds=1,child_cpu_seconds=1)
  return raw,r,d,t
 def checked(self,raw,r,d,t):return check_profile_records(raw,r,d,t,expected_profiler_sha256='synthetic_source_identity')
 def test_consistent_fixture_and_exact_protocol_match(self):
  p,c=self.checked(*self.fixture());cfg=dict(variant='author_default_DCNN_2L',cost_gate_passed=True,seed=7,batch_size=8,epochs=3,input_preprocessing=p['input_preprocessing']);self.assertEqual(match_protocol(cfg,p,c),cfg)
 def test_bool_cost_gate_cannot_rescue_changed_exact_configuration(self):
  p,c=self.checked(*self.fixture());cfg=dict(variant='author_default_DCNN_2L',cost_gate_passed=True,seed=7,batch_size=8,epochs=3,input_preprocessing=p['input_preprocessing'])
  for k,v in [('seed',8),('batch_size',16),('epochs',1),('input_preprocessing','changed'),('seed',True),('variant','shrunk_model')]:
   bad=dict(cfg);bad[k]=v
   with self.assertRaises(ValueError):match_protocol(bad,p,c)
 def test_mutated_declaration_or_termination_rejected(self):
  for part,key,value in [(2,'policy_sha256','wrong'),(2,'utility_sha256','other'),(2,'policy_fixed_before_timing',False),(2,'declared_unix',float('nan')),(3,'exit_code',1),(3,'wall_timeout',True),(3,'child_cpu_seconds',601),(3,'supervisor_wall_seconds',621)]:
   f=list(self.fixture());f[part][key]=value
   with self.assertRaises(ValueError):self.checked(*f)
 def test_incomplete_or_recomputed_cost_or_wrong_architecture_rejected(self):
  for key,value in [('total_parameters',1),('trainable_parameters',1),('source_actual_minibatch',16),('chosen_epochs',1),('training_seconds',[.2]),('memory_gate_passed',False),('real_pilot_started',True),('original_training_classweight_formula',False)]:
   f=list(self.fixture());f[1][key]=value
   with self.assertRaises(ValueError):self.checked(*f)
  f=list(self.fixture());f[1]['estimated_seconds_by_epoch_budget']['3']=1
  with self.assertRaises(ValueError):self.checked(*f)
 def test_resource_closed_result_refused_even_if_internally_consistent(self):
  raw,r,d,t=self.fixture();p=json.loads(raw);p['budget_seconds']=1;raw=json.dumps(p).encode();d['policy_sha256']=hashlib.sha256(raw).hexdigest();r['policy']=p;r.update(select(p,**{k:r[k] for k in ['training_seconds','evaluation_seconds','keras_save_seconds','snapshot_save_seconds','max_RSS_KiB']}))
  with self.assertRaisesRegex(ValueError,'did not pass'):self.checked(raw,r,d,t)
 def test_active_current_refusal_precedes_every_evidence_or_policy_read(self):
  with tempfile.TemporaryDirectory() as d,patch('cost_evidence.read',side_effect=AssertionError('Must not read evidence')),patch.object(Path,'read_bytes',side_effect=AssertionError('Must not read policy')):
   with self.assertRaisesRegex(RuntimeError,'audit not complete'):require_cost_and_registration('no_config','no_marker',d,'no_cost_workspace')
 def test_mocked_registered_fixture_binds_all_cost_evidence_bytes(self):
  # Files below contain invented unit-test metadata, not actual measurements.
  # Both prior closure and registration are explicitly mocked for this test.
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);profile=root/'profile01';term_dir=root/'supervisor01';profile.mkdir();term_dir.mkdir();prior=root/'synthetic_prior';raw,r,d,t=self.fixture();p=json.loads(raw);p['prior_workspace']=str(prior);raw=json.dumps(p).encode();r['policy']=p;d['policy_sha256']=hashlib.sha256(raw).hexdigest();d['utility_sha256']=sha(TOOLS/'profile_cost.py')
   (root/'POLICY.json').write_bytes(raw);(profile/'POLICY.json').write_bytes(raw);(root/'profile_cost.py').write_bytes((TOOLS/'profile_cost.py').read_bytes());(root/'COST_HASHES.json').write_text(json.dumps({n:sha(root/n) for n in ['POLICY.json','profile_cost.py']}))
   for name,obj in [('COST_CHECK.json',r),('DECLARATION_BEFORE_TIMING.json',d)]: (profile/name).write_text(json.dumps(obj))
   (term_dir/'TERMINATION.json').write_text(json.dumps(t))
   cfg=dict(variant='author_default_DCNN_2L',cost_gate_passed=True,seed=7,batch_size=8,epochs=3,input_preprocessing=p['input_preprocessing'],cost_evidence_sha256={str(path):sha(path) for path in [root/'POLICY.json',root/'COST_HASHES.json',profile/'COST_CHECK.json',profile/'DECLARATION_BEFORE_TIMING.json',term_dir/'TERMINATION.json']})
   with patch('cost_evidence.prior_closed',return_value=True),patch('cost_evidence.real_entry_ready',return_value=cfg):
    self.assertEqual(require_cost_and_registration('fixture','fixture',prior,root),cfg)
    cfg['cost_evidence_sha256'][str(profile/'COST_CHECK.json')]='wrong'
    with self.assertRaisesRegex(ValueError,'registered protocol'):require_cost_and_registration('fixture','fixture',prior,root)
if __name__=='__main__':unittest.main()
