"""Predeclared complete t-LeNet1000/B256 resource bound; no reduced fallback."""
import math
def select(p,training,evaluation,native_save,full_save,rss):
 assert (p['epochs'],p['batch_size'],p['seed'])==(1000,256,20261029)
 assert p['budget_seconds']==43200 and p['memory_limit_KiB']==14680064
 assert [len(x) for x in [training,evaluation,native_save,full_save]]==[8,4,3,3]
 for v in [training,evaluation,native_save,full_save]:assert all(type(x) in [int,float] and math.isfinite(x) and x>0 for x in v)
 assert type(rss) in [int,float] and math.isfinite(rss) and rss>0
 estimate=2*(184000*max(training)+40087*max(evaluation)+1001*max(native_save)+2004*max(full_save))+120
 memory=rss<=p['memory_limit_KiB'];passed=memory and estimate<=p['budget_seconds']
 return dict(state='pass' if passed else 'resource_closed_exact_policy',resource_gate_passed=passed,epochs=1000,batch_size=256,training_batches=184,validation_batches=40,diagnostic_batches=47,estimated_seconds=estimate,estimated_hours=estimate/3600,memory_gate_passed=memory,formula='2*(184000*maxtrain+40087*maxeval+1001*maxnative+2004*maxfullstate)+120',worst_case_best_last_fullstate_every_epoch_included=True,no_smaller_epoch_batch_fallback=True,accuracy_measured=False)
