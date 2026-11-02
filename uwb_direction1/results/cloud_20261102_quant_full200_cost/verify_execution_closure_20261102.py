"""Read-only verification of retained terminal evidence; never repeats the screen."""
from pathlib import Path
import json,hashlib,datetime,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B));from guard import ready,sha
ready()
t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
assert t['exit_code']==-15 and t['wall_timeout'] and t['resource_stop_reason']=='wall_resource_limit' and t['cgroup_oom_kill_delta']==0
assert t['supervisor_wall_seconds']>=3600 and t['child_cpu_seconds']>0 and t['peak_observed_RSS_KiB']<12582912
assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
assert not Path(f'/proc/{t["pid"]}').exists() and not (B/'profile01/COST_RESULT.json').exists() and not (B/'VERIFIED_COST_RESULT.json').exists()
r=dict(state='verified_full_QUANT200_onehour_synthetic_screen_incomplete_wall_closed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],termination=t,termination_sha256=sha(B/'supervisor01/TERMINATION.json'),full_fit_completed=False,full12h_estimate_available=False,resource_gate_passed=False,no_accuracy_measured=True,real_UWB_values_labels_fit=False,no_old_screen_retry=True,no_feature_tree_seed_precision_view_or_role_rescue=True,no_method_family_accuracy_rejection=True,next_action='implement_and_register_paper_scale_DrCIF500_native_transfer_cost')
(B/'VERIFIED_EXECUTION_CLOSURE_20261102.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
