"""Metadata and byte guards before NumPy import or UWB decode."""
from pathlib import Path
import hashlib,importlib.util,json
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def fixed(c):
 if (c['epochs'],c['batch_size'],c['seed'],c['threads'],c['parameters'],c['trainable_parameters'])!=(250,128,20261029,1,860364,859340):raise ValueError('Changed fixed method/budget')
 if c['counts']!={'train':4700,'validation':1000,'diagnostic':1200} or c['training_users']!=[3,4,5,6] or c['validation_user']!=2 or c['diagnostic_user']!=1:raise ValueError('Changed fixed source roles')
 if c['users7_8_labels_used'] or not c['cost_gate_passed'] or c['cost_estimate_seconds']>43200 or c['max_training_updates']!=9250 or c['max_training_appearances']!=1175000:raise ValueError('Absent cost or wrong scope')
 if c['gate']!={'minimum_gain_over_fixed_reference':0.005,'tolerance':1e-12,'pass_route':'separately_registered_source6_confirmation','fail_route':'close_exact_full250_B128_seed_configuration_no_rescue'}:raise ValueError('Changed predeclared accuracy gate')
 return c
def ready(base=B,require_marker=True):
 base=Path(base);c=fixed(read(base/'PROTOCOL.json'))
 if require_marker:
  m=read(base/'REMOTE_PREREGISTRATION.json')
  if m.get('state')!='remote_UWB_preregistration_exact_readback_verified' or m.get('protocol_sha256')!=sha(base/'PROTOCOL.json') or m.get('frozen_sha256')!=sha(base/'FROZEN_HASHES.json'):raise RuntimeError('No matching exact remote registration')
 for n,d in read(base/'FROZEN_HASHES.json').items():
  if sha(base/n)!=d:raise RuntimeError('Changed frozen prerequisite: '+n)
 cost=Path(c['cost_workspace'])
 for n,d in c['cost_evidence'].items():
  if sha(n)!=d:raise RuntimeError('Changed actual cost evidence')
 result=read(cost/'VERIFIED_COST_RESULT.json');raw=read(cost/'profile01/COST_CHECK.json');term=read(cost/'supervisor01/TERMINATION.json')
 if not result['resource_gate_passed'] or not result['profile_completed_successfully'] or result['accuracy_measured'] or result['real_UWB_values_labels_fit'] or term['exit_code']!=0 or term['wall_timeout']:raise RuntimeError('No successful bounded cost pass')
 if raw['policy']['epochs']!=250 or raw['policy']['batch_size']!=128 or raw['policy']['seed']!=20261029:raise RuntimeError('Cost identity mismatch')
 spec=importlib.util.spec_from_file_location('MLSTM_previous_closure',cost/'closure.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.check()
 e=Path(c['engine_preparation'])
 if sha(e/'PREPARATION_HASHES.json')!=c['engine_hashes_sha256']:raise RuntimeError('Changed engine evidence')
 for n,d in read(e/'PREPARATION_HASHES.json')['files'].items():
  if sha(e/n)!=d:raise RuntimeError('Changed checked engine file')
 return c
def require_training_complete(base=B):
 base=Path(base);out=base/'run01';s=read(out/'STATE.json');t=read(base/'supervisor01/TERMINATION.json')
 if s.get('state')!='complete' or s.get('pid') is not None or not s.get('all_predictions_frozen') or s['diagnostic_labels_read'] or s['users7_8_labels_used']:raise RuntimeError('Producer active/incomplete')
 if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Unsuccessful training termination')
 d=read(out/'TRAINING_COMPLETE.json')
 if d['epochs']!=250 or d['steps']!=9250 or not 1<=d['best_epoch']<=250:raise RuntimeError('Incomplete fixed training budget')
 for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]:
  if sha(out/name)!=d[key]:raise RuntimeError('Changed checkpoint')
 f=read(out/'ALL_PREDICTIONS_FROZEN.json')
 if set(f)!={'validation','diagnostic'}:raise RuntimeError('Missing dual freeze')
 for role,digest in f.items():
  if sha(out/(role+'_predictions_unscored.npz'))!=digest:raise RuntimeError('Changed frozen predictions')
 return t
