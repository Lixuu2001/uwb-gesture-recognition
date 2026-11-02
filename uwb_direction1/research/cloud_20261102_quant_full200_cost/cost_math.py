import math
def select(p,fit,evaluation,saves,loads,rss,size,free):
    assert (p['trees'],p['features'],p['depth'],p['div'],p['max_features'],p['criterion'],p['seed'],p['threads'],p['synthetic_train_size'])==(200,539217,6,4,.1,'entropy',20261029,4,4700)
    assert len(evaluation)==len(saves)==len(loads)==3
    assert all(math.isfinite(v) and v>0 for v in [fit,rss]+evaluation+saves+loads)
    estimate=3*fit+2*2200*max(evaluation)/64+4*max(saves)+2*max(loads)+180
    memory=rss+2097152;disk=8*size+67108864
    return dict(resource_gate_passed=estimate<=43200 and 4*estimate<=172800 and memory<=14680064 and disk<=free,estimated_CPU_seconds=4*estimate,estimated_seconds=estimate,estimated_hours=estimate/3600,estimated_max_RSS_KiB=memory,required_checkpoint_workspace_bytes=disk,checkpoint_bytes=size,workspace_free_bytes=free,accuracy_measured=False,real_UWB_values_labels_fit=False,real_pilot_started=False,no_feature_tree_depth_seed_precision_view_or_role_rescue=True)
