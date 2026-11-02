import json
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes());assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,3,90,189]
 assert r['features']==512 and r['full_training_cases']==4700 and r['full_encoder_layers']==[2,2,2,2] and r['encoder_weights_sha256']==p['weights_sha256'] and r['full_encoder_frozen_unchanged'] and r['head_converged'] and r['head_n_iter']<10000 and r['loaded_head_coefficients_decisions_predictions_exact'] and not r['actual_UWB_values_labels_fit']
 for i,n in enumerate(r['head_checkpoint_sizes']):assert (B/f'profile01/head_synthetic_{i}.pkl').stat().st_size==n
 v=dict(state='independently_verified_full4700_official_ResNet18_frozen512_LinearSVC_C1_cost',**select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_head_save_seconds'],r['complete_head_reload_seconds'],max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']),max(r['head_checkpoint_sizes']),r['workspace_free_bytes']),registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],real_fit_authorized=False)
 v['next_action']='separately_register_real_runner_full_audit_private_encoder_head_backup' if v['resource_gate_passed'] else 'close_exact_frozen_ResNet18_ImageNet_C1_configuration_no_rescue';(B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
