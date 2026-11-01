"""Require preserved H source6 failure and actual Disjoint resource closure."""
from pathlib import Path
import hashlib,importlib.util,json
PREV=Path('/workspace/uwb-disjoint-formal-cost-20261101')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 pinned=read(Path(__file__).resolve().parent/'DEPENDENCIES.json')
 for name,digest in pinned['prior_closure_files'].items():
  if sha(Path(name))!=digest:raise RuntimeError('Changed prior closure dependency: '+name)
 spec=importlib.util.spec_from_file_location('mlstm_prior_source6',PREV/'source6_closure.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.prior_closed('/workspace/uwb-hinception-sixfold-20261031')
 verified=read(PREV/'VERIFIED_COST_RESULT.json');t=read(PREV/'supervisor01/TERMINATION.json');r=read(PREV/'profile01/COST_CHECK.json')
 if verified['state']!='verified_actual_synthetic_profile_resource_closed_exact_full500_B8_policy' or verified['prospective_full_run_resource_gate_passed'] or verified['accuracy_measured']:raise RuntimeError('Disjoint policy not closed')
 for path,key in [('profile01/COST_CHECK.json','result_sha256'),('profile01/DECLARATION_BEFORE_TIMING.json','declaration_sha256'),('supervisor01/TERMINATION.json','termination_sha256')]:
  if sha(PREV/path)!=verified[key]:raise RuntimeError('Changed Disjoint evidence')
 if t['exit_code']!=0 or t['wall_timeout'] or r['state']!='resource_closed_exact_policy' or r['chosen_epochs'] is not None or r['policy']['epoch_candidates']!=[500] or r['policy']['batch_size']!=8 or r['real_UWB_values_labels_or_fit']:raise RuntimeError('Wrong predecessor scope')
 return True
