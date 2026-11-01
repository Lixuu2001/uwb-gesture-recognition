"""Fixed full FCN2000/B16 conservative complete-trajectory resource bound."""
import math
def select(p,training,evaluation,native_save,full_save,rss):
 assert (p['epochs'],p['batch_size'],p['seed'])==(2000,16,20261029)
 assert p['budget_seconds']==43200 and p['memory_limit_KiB']==14680064
 assert [len(v) for v in [training,evaluation,native_save,full_save]]==[8,4,3,3]
 for v in [training,evaluation,native_save,full_save]:assert all(type(x) in [int,float] and math.isfinite(x) and x>0 for x in v)
 assert type(rss) in [int,float] and math.isfinite(rss) and rss>0
 estimate=2*(2000*294*max(training)+126138*max(evaluation)+2001*max(native_save)+4004*max(full_save))+120
 memory=rss<=p['memory_limit_KiB'];passed=memory and estimate<=p['budget_seconds']
 return {'state':'pass' if passed else 'resource_closed_exact_policy','resource_gate_passed':passed,'epochs':2000,'batch_size':16,'training_batches':294,'validation_batches':63,'diagnostic_batches':75,'estimated_seconds':estimate,'estimated_hours':estimate/3600,'memory_gate_passed':memory,'formula':'2*(2000*294*maxtrain+126138*maxeval+2001*maxnative+4004*maxfullstate)+120','worst_case_best_last_fullstate_every_epoch_included':True,'no_smaller_epoch_batch_fallback':True,'accuracy_measured':False}
