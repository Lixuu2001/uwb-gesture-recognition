"""Verify retained memory-stop evidence without repeating frozen work."""
from pathlib import Path
import json,datetime
from guard import B,ready,sha
ready();t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
assert t['exit_code']==-15 and t['resource_stop_reason']=='RSS_resource_limit' and not t['wall_timeout'] and t['peak_observed_RSS_KiB']>12582912 and t['cgroup_oom_kill_delta']==0
assert not Path(f'/proc/{t["pid"]}').exists() and not (B/'profile01/COST_RESULT.json').exists() and not (B/'profile01/RESOURCE_RESULT.json').exists()
assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
r=dict(state='verified_native_fullbatch_DrCIF500_float64_RSS_screen_closed_before_fit_complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],termination=t,termination_sha256=sha(B/'supervisor01/TERMINATION.json'),full_fit_completed=False,full12h_estimate_available=False,accuracy_measured=False,real_UWB_values_labels_fit=False,no_old_screen_retry=True,next_action='separate_output_equivalent_batched_native_periodogram_execution_check_before_new_cost_registration',scientific_trees_thresholds_intervals_seed_precision_views_unchanged=True)
(B/'VERIFIED_EXECUTION_CLOSURE_20261102.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
