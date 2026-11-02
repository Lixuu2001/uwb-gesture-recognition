import math
def select(p,fit,evaluation,saves,rss):
    assert (p['groups'],p['kernels_per_group'],p['features'],p['seed'],p['threads'],p['synthetic_train_size'])==(64,8,53824,20261029,1,4700)
    assert len(evaluation)==len(saves)==3
    assert all(math.isfinite(v) and v>0 for v in [fit,rss]+evaluation+saves)
    estimate=3*fit+2*2200*max(evaluation)/64+4*max(saves)+180
    memory=rss+2*1024*1024
    return dict(state='pass' if estimate<=43200 and memory<=14680064 else 'resource_closed_exact_policy',
        resource_gate_passed=estimate<=43200 and memory<=14680064,estimated_seconds=estimate,
        estimated_hours=estimate/3600,estimated_max_RSS_KiB=memory,
        formula='3*full4700x567x90_native_fit+2*2200*max_checked_eval64/64+4*maxsave+180',
        accuracy_measured=False,real_pilot_started=False,no_group_kernel_feature_seed_alpha_fallback=True)
