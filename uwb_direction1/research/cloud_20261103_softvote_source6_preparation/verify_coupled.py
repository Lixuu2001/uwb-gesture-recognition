"""Post-run independent receipt/trace verification; never repeats profiling."""
from pathlib import Path
import json,math,hashlib
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    read=lambda p:json.loads(Path(p).read_text())
    p=read(B/'SOURCE6_METHOD_LOCK.json');r=read(B/'profile01/RESULT.json');e=read(B/'EMPIRICAL_COST_INPUTS.json')
    assert r['exit_codes']==[0,0] and r['resource_stop_reason'] is None and r['cgroup_OOM_kill_delta']==0 and r['resource_gate_passed']
    assert r['wall_seconds']<=p['profile_wall_seconds'] and r['total_child_CPU_seconds']<=2*p['profile_each_CPU_seconds']
    assert r['peak_combined_RSS_KiB']<=p['combined_RSS_stop_KiB']
    for arm,threads in [('timemil',1),('shuffle',4)]:
        d=read(B/f'profile01/{arm}_RESULT.json')
        assert d['threads']==threads and d['updates']==6 and not d['new_UWB_fit'] and d['synthetic_only']
        assert len(d['times'])==6 and all(math.isfinite(x) and x>0 for x in d['times'])
        rows=[json.loads(x) for x in (B/f'profile01/{arm}.log').read_text().splitlines()]
        assert [x['update'] for x in rows]==list(range(1,7))
    root=Path('/workspace/uwb-timemil-20261029');a=read(root/'COMPLETE_RESULT_AUDIT.json');ep=root/'run02/timemil/epochs.jsonl'
    assert sha(ep)==a['artifact_hashes'][str(ep)]
    tm=max(json.loads(x)['epoch_seconds'] for x in ep.read_text().splitlines())
    sh=read('/workspace/uwb-shufflenetv2-x1-adam100-recovery45-20261103/audit_supervisor01/TERMINATION.json')
    wall=max(1.5*1.2*64*tm,1.5*1.2*sh['supervisor_wall_seconds'])+1.5*1.2*sh['supervisor_wall_seconds']+600
    cpu=1.5*1.2*(64*tm+2*sh['child_cpu_seconds'])+2400
    assert wall==e['parallel_pair_plus_Shuffle_replay_wall_upper'] and cpu==e['perfold_CPU_upper']
    assert wall<=p['per_fold_wall_seconds'] and cpu<=p['per_fold_CPU_seconds']
    v=read(B/'VERIFIED_COUPLED_PREFLIGHT.json')
    assert v['profile_result_sha256']==sha(B/'profile01/RESULT.json')
    assert v['source6_method_lock_sha256']==sha(B/'SOURCE6_METHOD_LOCK.json')
    assert v['projected_perfold_wall_seconds']==wall and v['projected_perfold_CPU_seconds']==cpu
    assert not v['new_UWB_fit_started'] and not v['full_future_training_time_guaranteed']
    print(json.dumps(dict(state='all_coupled_preflight_receipts_and_empirical_arithmetic_verified_without_reprofile',profile_sha256=sha(B/'profile01/RESULT.json'),real_training_started=False)))
if __name__=='__main__':main()
