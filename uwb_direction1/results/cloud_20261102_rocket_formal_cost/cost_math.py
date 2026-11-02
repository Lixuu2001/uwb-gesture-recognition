import math
def select(p,transform,ridge,saves,rss):
    assert (p['n_kernels'],p['seed'],p['threads'])==(10000,20261029,1)
    assert len(transform)==3 and len(saves)==3 and p['synthetic_train_size']==4700
    assert all(math.isfinite(v) and v>0 for v in transform+saves+[ridge,rss])
    estimate=2*(6900*max(transform)/64+ridge+3*max(saves))+120
    # Include 6900-case signed map working set, labels/IDs, producer/controller margin.
    memory=rss+2*1024*1024
    return dict(state='pass' if estimate<=43200 and memory<=14680064 else 'resource_closed_exact_policy',
        resource_gate_passed=estimate<=43200 and memory<=14680064,
        estimated_seconds=estimate,estimated_hours=estimate/3600,
        formula='2*(6900*max_transform64/64+full4700x20000_native_RidgeCV+3*maxsave)+120',
        estimated_max_RSS_KiB=memory,accuracy_measured=False,
        no_kernel_alpha_seed_fallback=True,real_pilot_started=False)
