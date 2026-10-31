"""Prospective full-default DCNN2L CPU math; no chosen policy/data/defaults."""
import math

def validate(p):
 required=['epoch_candidates','batch_size','seed','input_preprocessing','budget_seconds','memory_limit_KiB','prior_workspace']
 if not all(k in p for k in required):raise ValueError('Missing prospective policy fields')
 es=p['epoch_candidates']
 if not isinstance(es,list) or not es or any(type(e) is not int or e<1 for e in es) or len(set(es))!=len(es):raise ValueError('Need distinct ordered positive epoch candidates')
 if type(p['batch_size']) is not int or p['batch_size']<1 or type(p['seed']) is not int or not 0<=p['seed']<2**32:raise ValueError('Invalid explicit batch or seed')
 if not isinstance(p['input_preprocessing'],str) or not p['input_preprocessing'].strip():raise ValueError('Declare input adaptation before timing')
 for k,limit in [('budget_seconds',43200),('memory_limit_KiB',14*1024**2)]:
  if type(p[k]) not in [int,float] or not math.isfinite(p[k]) or not 0<p[k]<=limit:raise ValueError('Invalid bounded resources')
 if not isinstance(p['prior_workspace'],str) or not p['prior_workspace']:raise ValueError('Missing exact prior workspace')
 return p

def select(p,*,training_seconds,evaluation_seconds,keras_save_seconds,snapshot_save_seconds,max_RSS_KiB):
 validate(p)
 def maximum(vs):
  if not vs or any(type(x) not in [int,float] or not math.isfinite(x) or x<=0 for x in vs):raise ValueError('Invalid timing')
  return max(vs)
 train=maximum(training_seconds);evaluation=maximum(evaluation_seconds);ksave=maximum(keras_save_seconds);snapshot=maximum(snapshot_save_seconds)
 if type(max_RSS_KiB) not in [int,float] or not math.isfinite(max_RSS_KiB):raise ValueError('Invalid peak RSS')
 batch=int(min(4700/10,p['batch_size']));tb=math.ceil(4700/batch);vb=math.ceil(1000/batch);db=math.ceil(1200/batch)
 estimates={str(e):2*(e*tb*train+(e*vb+vb+db)*evaluation+(e+1)*ksave+(2*e+4)*snapshot)+120 for e in p['epoch_candidates']}
 memory_ok=0<max_RSS_KiB<=p['memory_limit_KiB'];chosen=next((e for e in p['epoch_candidates'] if memory_ok and estimates[str(e)]<=p['budget_seconds']),None)
 return dict(state='pass' if chosen is not None else 'resource_closed_exact_policy',chosen_epochs=chosen,requested_batch_size=p['batch_size'],source_actual_minibatch=batch,training_batches=tb,validation_batches=vb,diagnostic_batches=db,estimated_seconds_by_epoch_budget=estimates,memory_gate_passed=memory_ok,formula='2*(E*ceil4700B*maxtrain+(E*ceil1000B+ceil1000B+ceil1200B)*maxeval+(E+1)*maxkeras+(2E+4)*maxsnapshot)+120; B=int(min470,Brequested)',worst_case_best_and_last_fullstate_every_epoch_included=True,accuracy_measured=False)
