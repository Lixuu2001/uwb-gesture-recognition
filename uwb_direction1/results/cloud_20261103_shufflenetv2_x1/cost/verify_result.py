from pathlib import Path
import json,math
from guard import B,ready,sha
from cost_math import select
p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes())
assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
assert r['state']=='complete_full_ShuffleNetV2_x1_alllayer_SGDM40_B15_synthetic_cost' and r['no_actual_UWB_values_labels_read'] and r['full40_epochs_not_run']
calc=select(p,r['full_B15_update_seconds'],r['last_B5_update_seconds'],r['evaluation_B15_seconds'],r['full_checkpoint_save_seconds'],r['full_checkpoint_load_seconds'],r['full_checkpoint_bytes'],r['peak_RSS_KiB'],r['workspace_free_bytes'])
for k,v in calc.items():assert r[k]==v,k
assert all(x['full_state_exact'] and x['bytes']>10000000 and not (B/'profile01'/x['name']).exists() for x in r['disposable_checkpoint_records'])
out=dict(r,state='independently_verified_full_ShuffleNetV2_x1_SGDM40_B15_cost',profile_result_sha256=sha(B/'profile01/COST_RESULT.json'),supervisor_termination_sha256=sha(B/'supervisor01/TERMINATION.json'),verified_without_reprofiling=True,real_training_allowed_only_after_streaming_backup_tests_and_new_preregistration=True)
(B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['state','estimated_hours','resource_gate_passed','required_additional_workspace_bytes','verified_without_reprofiling']}))
