"""Fixed MLSTM-FCN250/B128 conservative full-trajectory resource estimate."""
import math
def select(policy,training,evaluation,native_save,full_save,RSS_KiB):
 assert policy['epochs']==250 and policy['batch_size']==128 and policy['seed']==20261029
 assert policy['budget_seconds']==43200 and policy['memory_limit_KiB']==14680064
 assert [len(v) for v in [training,evaluation,native_save,full_save]]==[8,4,3,3]
 for v in [training,evaluation,native_save,full_save]:assert all(type(x) in [float,int] and math.isfinite(x) and x>0 for x in v)
 assert type(RSS_KiB) in [int,float] and math.isfinite(RSS_KiB) and RSS_KiB>0
 E=250;tb=math.ceil(4700/128);vb=math.ceil(1000/128);db=math.ceil(1200/128)
 estimate=2*(E*tb*max(training)+(E*vb+vb+db)*max(evaluation)+(E+1)*max(native_save)+(2*E+4)*max(full_save))+120
 memory_ok=RSS_KiB<=policy['memory_limit_KiB'];passed=memory_ok and estimate<=policy['budget_seconds']
 return {'state':'pass' if passed else 'resource_closed_exact_policy','resource_gate_passed':passed,'epochs':250,'batch_size':128,'training_batches':tb,'validation_batches':vb,'diagnostic_batches':db,'estimated_seconds':estimate,'estimated_hours':estimate/3600,'memory_gate_passed':memory_ok,'formula':'2*(250*ceil4700/128*maxtrain+(250*ceil1000/128+ceil1000/128+ceil1200/128)*maxeval+251*maxnative+504*maxfullstate)+120','worst_case_best_and_last_fullstate_every_epoch_included':True,'no_smaller_epoch_fallback':True,'accuracy_measured':False}
