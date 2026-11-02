import math
def select(p,forward,outputs,states,rss):
 assert (p['samples'],p['batch_size'],p['seed'],p['threads'])==(100,64,20261029,1) and p['budget_seconds']==43200 and p['memory_limit_KiB']==14680064
 assert [len(x) for x in [forward,outputs,states]]==[8,3,3]
 for xs in [forward,outputs,states]:assert all(type(x) in [int,float] and math.isfinite(x) and x>0 for x in xs)
 assert type(rss) in [int,float] and math.isfinite(rss) and rss>0
 estimate=2*(3535*max(forward)+3*max(outputs)+3*max(states))+120;memory=rss<=14680064;passed=memory and estimate<=43200
 return dict(state='pass' if passed else 'resource_closed_exact_MC100_B64_policy',resource_gate_passed=passed,forward_batches_per_producer=3535,estimated_seconds=estimate,estimated_hours=estimate/3600,memory_gate_passed=memory,formula='2*(3535*maxforward+3*maxoutputsave+3*maxstatesave)+120',accuracy_measured=False,no_smaller_T_batch_or_rate_fallback=True)
