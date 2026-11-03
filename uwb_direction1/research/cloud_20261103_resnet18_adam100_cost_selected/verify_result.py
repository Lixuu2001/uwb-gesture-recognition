"""UNEXECUTED independent arithmetic/receipt check. Does not reprofile."""
from pathlib import Path
import json,math,re
from activation_guard import B,ready,read,sha,term_ok
from cost_math import estimate
def main():
    p=ready();r=read(B/'profile01/COST_RESULT.json')
    t=read(B/'supervisor01/TERMINATION.json')
    assert term_ok(t)
    assert t['cgroup_events_available'] is True
    assert t['supervisor_wall_seconds']<=p['profile_wall_seconds']+5
    assert t['child_cpu_seconds']<=p['profile_CPU_hard_seconds']
    assert t['peak_observed_RSS_KiB']<=p['RSS_stop_KiB']
    assert r['state']=='complete_synthetic_full_ResNet18_Adam100_B20_cost'
    assert r['actual_100epochs_not_run'] is True
    assert r['new_UWB_values_or_labels_fit'] is False
    calc=estimate(p,r['full_B20_update_seconds'],r['evaluation_B20_seconds'],
        r['full_checkpoint_save_seconds'],r['full_checkpoint_load_seconds'],
        r['model_only_save_seconds'],r['model_only_load_seconds'],
        r['complete_checkpoint_bytes'],r['model_only_checkpoint_bytes'],
        r['peak_RSS_KiB'],r['workspace_free_bytes'])
    for k,v in calc.items():assert r[k]==v,k
    rows=r['checkpoint_records']
    assert len(rows)==6 and len(set(x['name'] for x in rows))==6
    assert sorted(x['kind'] for x in rows)==['full']*3+['model']*3
    for x in rows:
        assert x['all_saved_state_exact'] is True
        assert type(x['bytes']) is int and x['bytes']>0
        assert re.fullmatch('[0-9a-f]{64}',x['sha256'])
        assert Path(x['name']).name==x['name']
        assert x['name'].startswith('disposable_synthetic_')
        assert not (B/'profile01'/x['name']).exists()
    result=dict(r,state='verified_full_ResNet18_Adam100_B20_synthetic_cost',
        original_result_sha256=sha(B/'profile01/COST_RESULT.json'),
        termination_sha256=sha(B/'supervisor01/TERMINATION.json'),
        verified_without_reprofiling=True,real_training_started=False)
    with (B/'VERIFIED_COST_RESULT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['state','resource_gate_passed',
        'estimated_wall_seconds','estimated_CPU_seconds',
        'required_additional_workspace_bytes','verified_without_reprofiling']}))
if __name__=='__main__':main()
