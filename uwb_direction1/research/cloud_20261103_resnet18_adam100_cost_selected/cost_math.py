"""Pure budget calculation (synthetic boundary tests executed) for the fixed complete 100-epoch trial."""
import math
def estimate(p,steps,evaluation,full_saves,full_loads,model_saves,model_loads,
             full_bytes,model_bytes,rss_KiB,free_bytes):
    assert (p['epochs'],p['batch_size'],p['training_cases'],p['validation_cases'],
            p['diagnostic_cases'],p['trainable_parameters'],p['seed'],p['threads'])==(
            100,20,4700,1000,1200,11182668,20261029,4)
    assert (p['lr'],p['betas'],p['eps'],p['weight_decay'],p['augmentation'],
            p['checkpoint_selection'],p['precision'])==(
            .001,[.9,.999],1e-8,0,False,'fixed_epoch100','float32')
    assert len(steps)==6
    assert all(len(x)==3 for x in (evaluation,full_saves,full_loads,model_saves,model_loads))
    values=[v for x in (steps,evaluation,full_saves,full_loads,model_saves,model_loads) for v in x]
    assert all(math.isfinite(float(v)) and float(v)>0 for v in values)
    assert all(type(v) is int and v>0 for v in (full_bytes,model_bytes,rss_KiB,free_bytes))
    updates=p['epochs']*math.ceil(p['training_cases']/p['batch_size'])
    val=p['epochs']*math.ceil(p['validation_cases']/p['batch_size'])
    held=math.ceil((p['validation_cases']+p['diagnostic_cases'])/p['batch_size'])
    assert updates==23500 and val==5000 and held==110
    # Producer saves complete resume state at all 100 epoch boundaries.
    seconds=(2*updates*max(steps)+2*val*max(evaluation)+2*held*max(evaluation)
             +104*max(full_saves)+6*max(full_loads)
             +4*max(model_saves)+4*max(model_loads)+180)
    memory=rss_KiB+p['map_margin_KiB']
    archive_upper=math.ceil((p['initial_encoder_bytes']+full_bytes+model_bytes
                            +p['metadata_upper_bytes'])*1.01)+1048576
    persistent=full_bytes+model_bytes+65536
    checkpoint_peak=persistent+max(full_bytes,model_bytes)
    archive_peak=persistent+archive_upper+p['maximum_chunk_pipelines']*p['chunk_bytes']
    disk=max(checkpoint_peak,archive_peak)+p['workspace_margin_bytes']
    passed=(seconds<=p['budget_wall_seconds']
            and 4*seconds<=p['budget_CPU_seconds']
            and memory<=p['memory_budget_KiB'] and disk<=free_bytes)
    return dict(resource_gate_passed=passed,estimated_wall_seconds=seconds,
                estimated_CPU_seconds=4*seconds,estimated_RSS_KiB=memory,
                required_additional_workspace_bytes=disk,
                stream_archive_upper_bytes=archive_upper,
                complete_checkpoint_bytes=full_bytes,model_only_checkpoint_bytes=model_bytes,
                timing_observation_count=len(values),full_training_updates=updates,
                per_run_validation_batches=val,held_batches=held,
                fresh_full_method_no_scale_or_epoch_rescue=True,
                new_UWB_fit_started=False,accuracy_measured=False)
