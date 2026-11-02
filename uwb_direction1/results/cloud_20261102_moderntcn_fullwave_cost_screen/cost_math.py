import math
def evaluate(policy,training_wall,training_cpu,eval_wall,eval_cpu,rss):
    assert (policy['epochs'],policy['batch_size'],policy['threads'],policy['variables'],policy['seed'])==(100,32,4,567,20261029)
    assert [len(x) for x in [training_wall,training_cpu,eval_wall,eval_cpu]]==[3,3,3,3]
    assert all(math.isfinite(x) and x>0 for xs in [training_wall,training_cpu,eval_wall,eval_cpu,[rss]] for x in xs)
    wall=2*(14700*max(training_wall)+6470*max(eval_wall))+120
    cpu=2*(14700*max(training_cpu)+6470*max(eval_cpu))+480
    memory=rss+2097152
    rejected=wall>43200 or cpu>172800 or memory>14680064
    return dict(state='resource_closed_exact_full_waveGesture_policy' if rejected else 'inconclusive_requires_complete_checkpoint_storage_and_IO_cost',
        exact_policy_resource_rejected=rejected,resource_gate_passed=False,
        estimated_training_and_eval_wall_seconds=wall,estimated_training_and_eval_CPU_seconds=cpu,
        estimated_RSS_with_map_margin_KiB=memory,checkpoint_storage_and_IO_measured=False,
        accuracy_measured=False,real_UWB_values_labels_fit=False,real_pilot_started=False,
        formula_wall='2*(14700*max_steady_train_wall+6470*max_eval_wall)+120',
        formula_CPU='2*(14700*max_steady_train_CPU+6470*max_eval_CPU)+480')
