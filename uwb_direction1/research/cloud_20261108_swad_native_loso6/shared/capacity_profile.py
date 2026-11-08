"""Single fixed integrated calculation; reuse the core measurement without retiming."""
from pathlib import Path
from common import PHASE as P,read,write,sha

def main():
 assert not(P/'RESOURCE_GATE.json').exists()
 spec=read(P/'INTEGRATED_RESOURCE_PROTOCOL.json');locked=read(P/'INTEGRATION_MEASUREMENT_INPUTS.json')
 for n,h in locked['sources'].items():assert sha(P/n)==h,n
 core=P/'core_preparation/GENERATED_CORE_RESULT.json';provenance=read(P/'core_preparation/CORE_EXECUTION_PROVENANCE.json')
 assert sha(core)==provenance['result_sha256'] and sha(P/'swad_core.py')==provenance['sources']['swad_core.py']
 base=read(core)['profile'];assert base['gate_passed'] and base['projected_core_based_full6_reserve_seconds']==spec['core_estimate_seconds']
 assert read(P/'LIFECYCLE_TEST_RESULT.json')['state'].startswith('pass_') and read(P/'BN_SYNTHETIC_RESULT.json')['state'].startswith('pass_')
 inputs={};maxima={}
 for name,key in [('CHECKPOINT_COST.json','checkpoint'),('STEP_DIGEST_COST.json','step_digest')]:
  paths=sorted((P/'generated_lifecycle_logs').rglob(name));assert len(paths)==2
  values=[a for path in paths for a in read(path)];assert len(values)==(130 if key=='checkpoint' else 128)
  maxima[key]=max(max(a['wall'],a['CPU']) for a in values);inputs.update({str(path.relative_to(P)):sha(path) for path in paths})
 bn=read(P/'BN_SYNTHETIC_RESULT.json');inputs['BN_SYNTHETIC_RESULT.json']=sha(P/'BN_SYNTHETIC_RESULT.json')
 for key,field in [('BN','production_full_B64_BN_batch_timings'),('eval','production_full_B64_eval_batch_timings')]:
  values=bn[field];assert values and all(a['n']==64 for a in values);maxima[key]=max(max(a['wall'],a['CPU']) for a in values)
 additive=maxima['checkpoint']*spec['full_state_checkpoint_count']+maxima['step_digest']*spec['actual_main_updates']+maxima['BN']*spec['BN_source_batches_all6']+maxima['eval']*spec['output_inference_batches_upper']
 reserve=spec['core_estimate_seconds']-spec['core_reserved_extra_seconds']+max(spec['core_reserved_extra_seconds'],1.25*additive)
 result=dict(state='single_fixed_core_reuse_plus_measured_integrated_overhead_gate',gate_passed=reserve<=spec['main_cap'],reserve_seconds=reserve,remaining_seconds_to_fixed_cap=spec['main_cap']-reserve,SWAD_native_ratio=base['conservative_ratio'],maxima_seconds=maxima,unmultiplied_integrated_overhead_seconds=additive,inputs=inputs,measurement_inputs_sha256=sha(P/'INTEGRATION_MEASUREMENT_INPUTS.json'),resource_protocol_sha256=sha(P/'INTEGRATED_RESOURCE_PROTOCOL.json'),core_measurement_repeated=False,actual_UWB_values_or_labels_or_IDs_opened=False,new_UWB_fit_started=False,formula=spec['formula'],estimate_is_not_a_runtime_guarantee=True,actual_phase_wall_and_CPU_supervision_remains_43200=True)
 write(P/'RESOURCE_GATE.json',result);print(result)
if __name__=='__main__':main()
