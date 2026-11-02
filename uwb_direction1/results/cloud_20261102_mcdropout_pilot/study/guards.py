from pathlib import Path
import json,hashlib,importlib.util
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def fixed(c):
 if (c['samples'],c['batch_size'],c['seed'],c['threads'],c['parameters'])!=(100,64,20261029,1,850973):raise ValueError('Changed MC configuration')
 if c['counts']!={'validation':1000,'diagnostic':1200} or c['validation_user']!=2 or c['diagnostic_user']!=1 or c['reused_training_users']!=[3,4,5,6]:raise ValueError('Changed roles')
 if c['new_training'] or c['users7_8_labels_used'] or c['dropout_rate']!=.2 or c['dropout_nodes']!=3 or c['link']!='mean_sigmoid' or c['input_shape']!=[90,567]:raise ValueError('Changed estimator/input')
 if c['selected_epoch']!=41 or c['selected_checkpoint_sha256']!='33af44c40fd933629ddc1e97a843943f6e3dc82a5a74bf1c5200db272c53980a':raise ValueError('Changed frozen selected model')
 if c['role_seed_offsets']!={'validation':101,'diagnostic':102} or not c['cost_gate_passed'] or c['cost_estimate_seconds']>43200:raise ValueError('Changed seed/cost policy')
 if c['reference_held1_macro']!=.9866666666666667 or c['reference_predictions_sha256']!='d55515802fc7f824a2c2d435a6a6c47e312f49e5a425d32eda5d3d81cfa96766':raise ValueError('Changed fixed reference')
 if c['gate']!={'minimum_gain_over_fixed_reference':.005,'tolerance':1e-12,'pass_route':'separately_registered_source6_confirmation','fail_route':'close_exact_MC100_B64_p02_seed_configuration_no_rescue'}:raise ValueError('Changed accuracy gate')
 return c
def ready(base=B,require_marker=True):
 base=Path(base);c=fixed(read(base/'PROTOCOL.json'))
 if require_marker:
  r=read(base/'REMOTE_PREREGISTRATION.json')
  if r.get('state')!='remote_UWB_preregistration_exact_readback_verified' or r.get('protocol_sha256')!=sha(base/'PROTOCOL.json') or r.get('frozen_sha256')!=sha(base/'FROZEN_HASHES.json'):raise RuntimeError('No matching remote real MC registration')
 for n,d in read(base/'FROZEN_HASHES.json').items():assert sha(base/n)==d,n
 for n,d in c['cost_evidence'].items():assert sha(n)==d,n
 for n in ['selected_checkpoint','reference_predictions','source_manifest','source_maps']:assert sha(c[n])==c[n+'_sha256'],n
 cost=Path(c['cost_workspace']);v=read(cost/'VERIFIED_COST_RESULT.json');t=read(cost/'supervisor01/TERMINATION.json');assert v['profile_completed_successfully'] and v['resource_gate_passed'] and not v['real_UWB_values_labels_fit'] and not v['trained_UWB_checkpoint_loaded'] and t['exit_code']==0 and not t['wall_timeout']
 spec=importlib.util.spec_from_file_location('MC100_cost_guard',cost/'guard.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);g.ready()
 old=read(c['reused_training_audit']);assert old['state']=='pass' and not old['users7_8_labels_used'];return c
def require_inference_complete(base=B):
 base=Path(base);out=base/'run01';s=read(out/'STATE.json');t=read(base/'supervisor01/TERMINATION.json')
 if s.get('state')!='complete' or s.get('pid') is not None or not s.get('all_predictions_frozen') or s['diagnostic_labels_read'] or s['users7_8_labels_used'] or t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('MC inference active/incomplete/failed')
 r=read(out/'INFERENCE_COMPLETE.json')
 if r['samples_per_role']!=100 or r['gradient_updates']!=0 or not r['baseline_same_ID_control_exact'] or not r['model_BN_buffers_and_global_RNG_unchanged']:raise RuntimeError('Wrong estimator completion')
 for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
 return t
