"""Verify actual preregistered MLSTM timing and accept either resource decision."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from cost_math import select
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 hashes=read(B/'COST_HASHES.json');marker=read(B/'REMOTE_PREREGISTRATION.json')
 assert marker['state']=='remote_cost_preregistration_exact_readback_verified' and marker['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for n,d in hashes.items():assert sha(B/n)==d,n
 p=read(B/'POLICY.json');r=read(B/'profile01/COST_CHECK.json');d=read(B/'profile01/DECLARATION_BEFORE_TIMING.json');t=read(B/'supervisor01/TERMINATION.json')
 assert (B/'profile01/POLICY.json').read_bytes()==(B/'POLICY.json').read_bytes()
 assert d['policy_fixed_before_timing'] and not d['real_UWB_values_labels_or_fit'] and d['policy_sha256']==sha(B/'POLICY.json') and d['profiler_sha256']==hashes['profile_cost.py'] and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 assert t['exit_code']==0 and not t['wall_timeout'] and 0<t['supervisor_wall_seconds']<=620 and 0<t['child_cpu_seconds']<=600
 assert r['policy']==p and r['full_documented_MLSTM_FCN8_128_256_128_SE16_Dropout08'] and r['total_parameters']==860364 and r['trainable_parameters']==859340 and r['LSTM_scan_steps']==567 and r['native_recurrent_gate']=='sigmoid' and not r['legacy_TF_numeric_parity']
 for n in ['synthetic_inverse_frequency_sample_weights','validation_unweighted','synthetic_fixture_only']:assert r[n]
 assert not r['real_UWB_values_labels_or_fit'] and not r['real_pilot_started']
 decision=select(p,r['training_seconds'],r['evaluation_seconds'],r['native_save_seconds'],r['fullstate_save_seconds'],r['max_RSS_KiB'])
 for n,v in decision.items():assert r[n]==v,n
 verified={'state':'verified_actual_full_MLSTM_FCN8_250_B128_synthetic_cost','registration_commit':marker['commit'],'policy_sha256':sha(B/'POLICY.json'),'profiler_sha256':hashes['profile_cost.py'],'result_sha256':sha(B/'profile01/COST_CHECK.json'),'declaration_sha256':sha(B/'profile01/DECLARATION_BEFORE_TIMING.json'),'termination_sha256':sha(B/'supervisor01/TERMINATION.json'),'profile_completed_successfully':True,'profile_exit_code':0,'resource_gate_passed':decision['resource_gate_passed'],'estimated_full250_seconds':decision['estimated_seconds'],'estimated_full250_hours':decision['estimated_hours'],'resource_budget_seconds':43200,'memory_gate_passed':decision['memory_gate_passed'],'max_RSS_KiB':r['max_RSS_KiB'],'next_action':'separately_register_actual_UWB_epoch_callback_engine_protocol_and_full_auditor' if decision['resource_gate_passed'] else 'close_exact_native_full250_B128_policy_without_smaller_budget_rescue','accuracy_measured':False,'real_UWB_values_labels_fit':False,'historical_TF_parity':False,'no_smaller_epoch_batch_fallback':True}
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(verified,indent=2)+'\n');print(json.dumps(verified))
if __name__=='__main__':main()
