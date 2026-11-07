"""Single fixed generated B64 resource measurement; no UWB input/target or model selection."""
from pathlib import Path
import json,time,resource,signal
import native_engine as e
P=Path(__file__).resolve().parent

def main():
    cfg=json.loads((P/'CAPACITY_PROTOCOL.json').read_text());assert not(P/'CAPACITY_PROFILE.json').exists()
    def stop(*_):raise TimeoutError('fixed generated profile resource cap')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600);resource.setrlimit(resource.RLIMIT_CPU,(600,605))
    t=e.torch;x=t.randn(64,90,567,generator=t.Generator().manual_seed(7821));y=t.arange(64)%12;domains=t.arange(64)//16
    rows=[]
    for arm in cfg['arm_order']:
        m,opt=e.make();m.train()
        if arm=='irm':opt.algorithm_updates=500;opt.inner.count=500
        for _ in range(3):e.step(m,opt,x,y,domains,arm)
        wall=time.monotonic();cpu=time.process_time()
        for _ in range(6):last=e.step(m,opt,x,y,domains,arm)
        rows.append(dict(arm=arm,updates=6,mean_wall=(time.monotonic()-wall)/6,mean_CPU=(time.process_time()-cpu)/6,last_generated_step=last))
    q=rows[1];estimate=1.5*(55552*max(q['mean_wall'],q['mean_CPU'])+1800)
    result={'state':'one_fixed_generated_native_IRM_resource_profile','rows':rows,'estimated_main_production_plus_fresh_replay_seconds':estimate,'estimated_hours':estimate/3600,'main_wall_CPU_cap_seconds':43200,'resource_gate_passed':estimate<=43200,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'real_UWB_samples_labels_or_IDs_read':False,'new_UWB_fit_started':False,'fitting_preregistered':False,'single_generated_profile_not_accuracy_or_speedup_evidence':True}
    assert result['max_RSS_KiB']<=12582912
    (P/'CAPACITY_PROFILE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
