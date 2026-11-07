"""One fixed script execution audit/profile; same math, complete model and capacity rule."""
from pathlib import Path
import copy,json,time,signal,resource
import native_engine as e
from script_equivalence import ScriptableNativeGAP
P=Path(__file__).resolve().parent

def main():
    t=e.torch;proof=json.loads((P/'SCRIPT_EQUIVALENCE.json').read_text());assert proof['gate_passed']
    assert not(P/'SCRIPT_CAPACITY_PROFILE.json').exists()
    def stop(*_):raise TimeoutError('fixed generated script profile cap')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600);resource.setrlimit(resource.RLIMIT_CPU,(600,605))
    x=t.randn(64,90,567,generator=t.Generator().manual_seed(7821));y=t.arange(64)%12;d=t.arange(64)//16
    eager,o=e.make();raw,so=e.make();script=t.jit.script(ScriptableNativeGAP(raw));eager.train();script.train()
    for k in range(7):
        before=t.get_rng_state().clone();e.step(eager,o,x,y,d);after=t.get_rng_state().clone()
        t.set_rng_state(before);e.step(script,so,x,y,d)
        assert t.equal(after,t.get_rng_state()) and e.semantic(e.capture(eager,o))==e.semantic(e.capture(script,so))
    (P/'SCRIPT_MULTI_UPDATE_EQUIVALENCE.json').write_text(json.dumps({'state':'pass_seven_actual_native_script_and_eager_updates_bitwise_model_BN_full_optimizer_slow6_and_allRNG','optimizer_updates_each':7,'post_profiling_executor_warmup_included':True,'real_UWB_values_read':False},indent=2)+'\n')
    rows=[];compilecost=[]
    for arm in ['native','irm']:
        m,opt=e.make();w=time.monotonic();c=time.process_time();m=t.jit.script(ScriptableNativeGAP(m));compilecost.append(max(time.monotonic()-w,time.process_time()-c));m.train()
        assert all(p.data_ptr()==q.data_ptr() for p,q in zip(m.parameters(),opt.params))
        if arm=='irm':opt.algorithm_updates=500;opt.inner.count=500
        for _ in range(3):e.step(m,opt,x,y,d,arm)
        w=time.monotonic();c=time.process_time()
        for _ in range(6):last=e.step(m,opt,x,y,d,arm)
        rows.append({'arm':arm,'mean_wall':(time.monotonic()-w)/6,'mean_CPU':(time.process_time()-c)/6,'last_generated_step':last})
    estimate=1.5*(55552*max(rows[1]['mean_wall'],rows[1]['mean_CPU'])+1800)+12*max(compilecost)
    result={'state':'one_fixed_fullnative_script_IRM_capacity_measurement','implementation':'torch.jit.script; zero architecture or precision changes','rows':rows,'estimated_main_production_plus_fresh_replay_seconds':estimate,'estimated_hours':estimate/3600,'compile12_worker_allowance_seconds':12*max(compilecost),'main_wall_CPU_cap_seconds':43200,'resource_gate_passed':estimate<=43200,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'real_UWB_samples_labels_or_IDs_read':False,'new_UWB_fit_started':False,'fitting_preregistered':False,'unoptimized_estimate_preserved':True}
    assert result['max_RSS_KiB']<=12582912
    (P/'SCRIPT_CAPACITY_PROFILE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
