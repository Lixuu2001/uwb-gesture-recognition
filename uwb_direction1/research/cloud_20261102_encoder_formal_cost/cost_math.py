"""Predeclared complete Encoder100/B12 resource bound; no reduced fallback."""
import math
def select(p,training,evaluation,native_save,full_save,rss):
 assert (p['epochs'],p['batch_size'],p['seed'])==(100,12,20261029)
 assert p['budget_seconds']==43200 and p['memory_limit_KiB']==14680064
 assert [len(x) for x in [training,evaluation,native_save,full_save]]==[8,4,3,3]
 for v in [training,evaluation,native_save,full_save]:assert all(type(x) in [int,float] and math.isfinite(x) and x>0 for x in v)
 assert type(rss) in [int,float] and math.isfinite(rss) and rss>0
 estimate=2*(100*392*max(training)+8584*max(evaluation)+101*max(native_save)+204*max(full_save))+120
 memory=rss<=p['memory_limit_KiB'];passed=memory and estimate<=p['budget_seconds']
 return dict(state='pass' if passed else 'resource_closed_exact_policy',resource_gate_passed=passed,epochs=100,batch_size=12,training_batches=392,validation_batches=84,diagnostic_batches=100,estimated_seconds=estimate,estimated_hours=estimate/3600,memory_gate_passed=memory,formula='2*(100*392*maxtrain+8584*maxeval+101*maxnative+204*maxfullstate)+120',worst_case_best_last_fullstate_every_epoch_included=True,no_smaller_epoch_batch_fallback=True,accuracy_measured=False)
