import math
def select(p,training,evaluation,native,full,rss):
 assert (p['epochs'],p['batch_size'],p['seed'])==(20,64,20261029)
 assert p['budget_seconds']==43200 and p['memory_limit_KiB']==14680064
 assert [len(x) for x in [training,evaluation,native,full]]==[8,4,3,3]
 for xs in [training,evaluation,native,full]:assert all(type(x) in [int,float] and math.isfinite(x) and x>0 for x in xs)
 assert type(rss) in [int,float] and math.isfinite(rss) and rss>0
 estimate=2*(1480*max(training)+355*max(evaluation)+2*max(native)+24*max(full))+120
 memory=rss<=p['memory_limit_KiB'];passed=memory and estimate<=p['budget_seconds']
 return dict(state='pass' if passed else 'resource_closed_exact_policy',resource_gate_passed=passed,epochs=20,batch_size=64,training_batches=74,validation_batches=16,diagnostic_batches=19,estimated_seconds=estimate,estimated_hours=estimate/3600,memory_gate_passed=memory,formula='2*(1480*maxtrain+355*maxeval+2*maxnative+24*maxfullstate)+120',no_smaller_epoch_batch_fallback=True,accuracy_measured=False)
