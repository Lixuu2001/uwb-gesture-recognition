"""Independent supervisor-bound cost gate; never reads real UWB arrays."""
from pathlib import Path
import hashlib,json
from guard import B,ready
from cost_math import decision
def main():
    policy=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes())
    assert policy==json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
    assert t['exit_code']==0 and not t['resource_stop_reason'] and not t['wall_timeout'] and t['cgroup_oom_kill_delta']==0
    assert not r['actual_UWB_values_labels_fit'] and not r['accuracy_measured'] and not r['scientific_negative']
    assert r['parameters']==2621070 and r['input_shape']==[32,90,567,1] and r['threads']==4
    assert r['checkpoint_predictions_and_Adam_state_exact']
    peak=max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB'])
    v=decision(r['training_update_seconds'],r['evaluation_batch_seconds'],r['checkpoint_save_seconds'],r['checkpoint_reload_seconds'],peak,max(r['checkpoint_sizes']),r['workspace_free_bytes'])
    for i,n in enumerate(r['checkpoint_sizes']):assert (B/f'profile01/synthetic_full_{i}.keras').stat().st_size==n
    result=dict(state='independently_verified_full_XCM128_100E_B32_resource_cost',**v,profile_peak_RSS_KiB=r['peak_RSS_KiB'],supervisor_peak_RSS_KiB=t['peak_observed_RSS_KiB'],registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],cost_hashes_sha256=hashlib.sha256((B/'COST_HASHES.json').read_bytes()).hexdigest(),actual_UWB_values_labels_fit=False,accuracy_measured=False,scientific_negative=False,real_fit_authorized=False,next_action='register_real_runner_full_auditor_and_private_model_preservation_before_real_fit' if v['resource_gate_passed'] else 'close_exact_full_XCM128_100E_B32_CPU4_resource_configuration_no_rescue')
    (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
