from pathlib import Path
import hashlib,json,importlib.util
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def fixed(c):
 if (c['epochs'],c['batch_size'],c['seed'],c['threads'],c['parameters'])!=(10,32,20261029,1,628752):raise ValueError('Changed fixed policy')
 if c['counts']!={'train':4700,'validation':1000,'diagnostic':1200} or c['training_users']!=[3,4,5,6] or c['validation_user']!=2 or c['diagnostic_user']!=1:raise ValueError('Changed roles')
 if c['users7_8_labels_used'] or c['loss']!='CrossEntropy' or c['optimizer']!='RAdam.0001' or c['patience']!=3 or c['gradient_clip']!=4. or not c['train_shuffle']:raise ValueError('Changed source semantics')
 if c['input_shape']!=[90,567] or c['max_training_updates']!=1470 or c['max_training_appearances']!=47000 or not c['cost_gate_passed'] or c['cost_estimate_seconds']>43200:raise ValueError('Changed budget/cost')
 if c['gate']!={'minimum_gain_over_fixed_reference':.005,'tolerance':1e-12,'pass_route':'separately_registered_source6_confirmation','fail_route':'close_exact_DLinear10_B32_patience3_seed_configuration_no_rescue'}:raise ValueError('Changed gate')
 if c['reference_held1_macro']!=.9866666666666667 or c['reference_predictions_sha256']!='d55515802fc7f824a2c2d435a6a6c47e312f49e5a425d32eda5d3d81cfa96766':raise ValueError('Changed reference')
 return c
def ready(base=B,require_marker=True):
 base=Path(base);c=fixed(read(base/'PROTOCOL.json'))
 if require_marker:
  m=read(base/'REMOTE_PREREGISTRATION.json')
  if m.get('state')!='remote_UWB_preregistration_exact_readback_verified' or m.get('protocol_sha256')!=sha(base/'PROTOCOL.json') or m.get('frozen_sha256')!=sha(base/'FROZEN_HASHES.json'):raise RuntimeError('No matching remote registration')
 for n,d in read(base/'FROZEN_HASHES.json').items():
  if sha(base/n)!=d:raise RuntimeError('Changed frozen prerequisite '+n)
 for n,d in c['cost_evidence'].items():assert sha(n)==d,n
 cost=Path(c['cost_workspace']);v=read(cost/'VERIFIED_COST_RESULT.json');t=read(cost/'supervisor01/TERMINATION.json')
 assert v['profile_completed_successfully'] and v['resource_gate_passed'] and not v['accuracy_measured'] and not v['real_UWB_values_labels_fit'] and t['exit_code']==0 and not t['wall_timeout']
 spec=importlib.util.spec_from_file_location('DLinear_cost_guard',cost/'guard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ready()
 return c
def require_training_complete(base=B):
 base=Path(base);out=base/'run01';s=read(out/'STATE.json');t=read(base/'supervisor01/TERMINATION.json')
 if s.get('state')!='complete' or s.get('pid') is not None or not s.get('all_predictions_frozen') or s['diagnostic_labels_read'] or s['users7_8_labels_used'] or t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Training active/incomplete/failed')
 done=read(out/'TRAINING_COMPLETE.json')
 if not (1<=done['epochs']<=10 and done['steps']==147*done['epochs'] and 1<=done['selected_epoch']<=done['epochs'] and (done['epochs']==10 or done['early_stop'])):raise RuntimeError('Wrong final source budget')
 for n,d in read(out/'FILES.json').items():
  if sha(out/n)!=d:raise RuntimeError('Changed closed producer '+n)
 return t
