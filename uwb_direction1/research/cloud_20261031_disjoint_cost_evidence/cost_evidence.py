"""GPL-3.0-only. Check actual future cost provenance before UWB entry.

No real policy, budget, seed, runtime profile or UWB configuration is selected.
Synthetic metadata fixtures in tests must never be treated as measurements.
"""
from pathlib import Path
import hashlib,json,math,sys
TOOLS=Path('/workspace/uwb-disjoint-cost-tools-20261031')
SOURCE=Path('/workspace/uwb-disjoint-source-probe-20261031')
sys.path.insert(0,str(TOOLS));sys.path.insert(0,str(SOURCE))
from cost_tools import validate,select
from readiness_guards import prior_closed,real_entry_ready

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_profile_records(policy_raw,result,declaration,termination,*,expected_profiler_sha256):
 """Pure evidence consistency checker; not an execution/timing claim."""
 policy=validate(json.loads(policy_raw))
 if declaration.get('policy_sha256')!=hashlib.sha256(policy_raw).hexdigest() or declaration.get('utility_sha256')!=expected_profiler_sha256 or declaration.get('policy_fixed_before_timing') is not True or declaration.get('real_UWB_values_labels_or_fit') is not False:raise ValueError('Wrong profile declaration or source identity')
 if type(declaration.get('declared_unix')) not in [int,float] or not math.isfinite(declaration['declared_unix']) or declaration['declared_unix']<=0:raise ValueError('Invalid declaration timestamp')
 if type(termination.get('exit_code')) is not int or termination['exit_code']!=0 or termination.get('wall_timeout') is not False:raise ValueError('Missing successful bounded profile termination')
 for key,limit in [('supervisor_wall_seconds',620),('child_cpu_seconds',600)]:
  value=termination.get(key)
  if type(value) not in [int,float] or not math.isfinite(value) or not 0<value<=limit:raise ValueError('Profile exceeded independent process bounds')
 if result.get('policy')!=policy or result.get('full_author_default_DCNN2L_architecture') is not True or result.get('total_parameters')!=2587468 or result.get('trainable_parameters')!=2586956:raise ValueError('Wrong full architecture or prospective policy')
 for key in ['original_training_classweight_formula','validation_unweighted','synthetic_fixture_only','preprocessing_choice_recorded_but_no_UWB_preprocessing_executed']:
  if result.get(key) is not True:raise ValueError('Missing explicit profile semantics: '+key)
 if result.get('real_UWB_values_labels_or_fit') is not False or result.get('real_pilot_started') is not False:raise ValueError('Profile unexpectedly accessed real data or launched a pilot')
 lengths={'training_seconds':8,'evaluation_seconds':4,'keras_save_seconds':3,'snapshot_save_seconds':3}
 for key,n in lengths.items():
  if not isinstance(result.get(key),list) or len(result[key])!=n:raise ValueError('Incomplete timing vector: '+key)
 recomputed=select(policy,training_seconds=result['training_seconds'],evaluation_seconds=result['evaluation_seconds'],keras_save_seconds=result['keras_save_seconds'],snapshot_save_seconds=result['snapshot_save_seconds'],max_RSS_KiB=result.get('max_RSS_KiB'))
 for key,value in recomputed.items():
  if result.get(key)!=value:raise ValueError('Cost arithmetic/result mismatch: '+key)
 if recomputed['state']!='pass' or recomputed['chosen_epochs'] is None:raise ValueError('Exact prospective resource policy did not pass')
 return policy,recomputed

def match_protocol(cfg,policy,decision):
 if cfg.get('variant')!='author_default_DCNN_2L' or cfg.get('cost_gate_passed') is not True:raise ValueError('Wrong registered model or absent cost gate')
 expected=dict(seed=policy['seed'],batch_size=policy['batch_size'],epochs=decision['chosen_epochs'],input_preprocessing=policy['input_preprocessing'])
 for key,value in expected.items():
  if type(cfg.get(key)) is not type(value) or cfg[key]!=value:raise ValueError('Protocol differs from exact cost choice: '+key)
 return cfg

def require_cost_and_registration(protocol,marker,prior,cost_workspace):
 # This first guard precedes policy/config/marker/cost reads, and every future
 # real-value loader must be called only after this entire function returns.
 prior_closed(prior)
 root=Path(cost_workspace);manifest=read(root/'COST_HASHES.json')
 if not isinstance(manifest,dict) or not manifest:raise ValueError('Missing actual cost dependency freeze')
 needed={'POLICY.json','profile_cost.py'}
 if not needed.issubset(manifest):raise ValueError('Missing frozen policy/profiler')
 for name,digest in manifest.items():
  if not isinstance(name,str) or not isinstance(digest,str) or sha(root/name)!=digest:raise ValueError('Changed cost dependency')
 if manifest['profile_cost.py']!=sha(TOOLS/'profile_cost.py'):raise ValueError('Different cost profiler from validated preparation')
 raw=(root/'POLICY.json').read_bytes();profile=root/'profile01'
 if (profile/'POLICY.json').read_bytes()!=raw:raise ValueError('Profile used different policy bytes')
 policy,decision=check_profile_records(raw,read(profile/'COST_CHECK.json'),read(profile/'DECLARATION_BEFORE_TIMING.json'),read(root/'supervisor01/TERMINATION.json'),expected_profiler_sha256=manifest['profile_cost.py'])
 if Path(policy['prior_workspace']).resolve()!=Path(prior).resolve():raise ValueError('Cost used a different prior closure')
 cfg=real_entry_ready(protocol,marker,prior)
 match_protocol(cfg,policy,decision)
 expected={str(root/'POLICY.json'):sha(root/'POLICY.json'),str(root/'COST_HASHES.json'):sha(root/'COST_HASHES.json'),str(profile/'COST_CHECK.json'):sha(profile/'COST_CHECK.json'),str(profile/'DECLARATION_BEFORE_TIMING.json'):sha(profile/'DECLARATION_BEFORE_TIMING.json'),str(root/'supervisor01/TERMINATION.json'):sha(root/'supervisor01/TERMINATION.json')}
 if cfg.get('cost_evidence_sha256')!=expected:raise ValueError('Exact cost evidence not covered by registered protocol bytes')
 return cfg
