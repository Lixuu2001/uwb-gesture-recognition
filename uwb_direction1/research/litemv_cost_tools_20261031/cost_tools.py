"""Prospective full-LITEMV cost arithmetic; no real policy defaults or data."""
import math
def validate(policy):
 required=['epoch_candidates','batch_size','seed','input_preprocessing','budget_seconds','memory_limit_KiB','prior_workspace']
 if not all(k in policy for k in required):raise ValueError('Missing prospective policy fields')
 epochs=policy['epoch_candidates']
 if not isinstance(epochs,list) or not epochs or any(type(x) is not int or x<=0 for x in epochs) or len(set(epochs))!=len(epochs):raise ValueError('Need a fixed ordered list of distinct positive epoch budgets')
 if type(policy['batch_size']) is not int or policy['batch_size']<=0 or type(policy['seed']) is not int or not 0<=policy['seed']<2**32:raise ValueError('Invalid fixed batch/seed')
 if policy['input_preprocessing'] not in ['explicit_RMS_adaptation','explicit_RMS_then_author_znormalization_adaptation']:raise ValueError('Input adaptation must be specified before timing')
 if not 0<policy['budget_seconds']<=43200 or not 0<policy['memory_limit_KiB']<=14*1024**2:raise ValueError('Invalid bounded CPU resource policy')
 if not isinstance(policy['prior_workspace'],str) or not policy['prior_workspace']:raise ValueError('Missing prior closure path')
 return policy
def select(policy,*,training_seconds,evaluation_seconds,keras_save_seconds,snapshot_save_seconds,max_RSS_KiB):
 validate(policy)
 def maximum(values):
  if not values or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Invalid timing measurement')
  return max(values)
 train=maximum(training_seconds);evaluation=maximum(evaluation_seconds);ksave=maximum(keras_save_seconds);snapshot=maximum(snapshot_save_seconds);batch=policy['batch_size'];tb=math.ceil(4700/batch);vb=math.ceil(1000/batch);db=math.ceil(1200/batch)
 estimates={str(ep):2*(ep*tb*train+(ep*vb+vb+db)*evaluation+(ep+1)*ksave+(ep+4)*snapshot)+120 for ep in policy['epoch_candidates']}
 memory_ok=0<max_RSS_KiB<=policy['memory_limit_KiB'];chosen=next((ep for ep in policy['epoch_candidates'] if memory_ok and estimates[str(ep)]<=policy['budget_seconds']),None)
 return dict(state='pass' if chosen is not None else 'resource_closed_exact_policy',chosen_epochs=chosen,estimated_seconds_by_epoch_budget=estimates,formula='2*(E*ceil4700B*maxtrain+(E*ceil1000B+ceil1000B+ceil1200B)*maxeval+(E+1)*maxkeras_save+(E+4)*maxfullsnapshot_save)+120',training_batches=tb,validation_batches=vb,diagnostic_batches=db,memory_gate_passed=memory_ok,accuracy_measured=False)
