"""ONE fixed generated fullnative Fishr cost gate. Three warmup/six active steps, no grid."""
import time,json,resource
from common import B,PHASE,ready,write,read
import engine as e

def main():
    ready();assert not(PHASE/'RESOURCE_GATE.json').exists(),'Single fixed profile; no rescue rerun'
    plan=read(PHASE/'BUDGET_PLAN.json');rng=e.np.random.default_rng(909);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12
    domains=e.np.repeat(e.c['train_users'],16);rows=[];whole=time.monotonic();cpu=time.process_time()
    for arm in ['native','Fishr']:
        m,o,g=e.make();m.train()
        for j in range(9):
            if arm=='Fishr' and j==3:o.algorithm_updates=1500;o.inner.count=1500
            w=time.monotonic();c=time.process_time()
            if arm=='native':
                o.zero_grad();z=m(e.torch.from_numpy(x));loss=e.native_bce(z,e.torch.as_tensor(y));loss.backward();e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step()
            else:
                r=e.step(m,o,x,y,1,domains)
                assert r['completed_updates_after']==(j+1 if j<3 else 1501+j-3)
                assert r['fishr_penalty_weight']==(0 if j<3 else 1000)
                assert r['optimizer_reset_this_update']==(j==3)
            rows.append(dict(arm=arm,step=j,wall=time.monotonic()-w,cpu=time.process_time()-c,active_penalty=arm=='Fishr' and j>=3,synthetic_counter_fixture1500=arm=='Fishr' and j==3))
    sums={arm:{k:sum(r[k] for r in rows if r['arm']==arm) for k in ['wall','cpu']} for arm in ['native','Fishr']}
    ratio=max(sums['Fishr'][k]/sums['native'][k] for k in ['wall','cpu']);reserve=1.25*31189.589142517*max(1.,ratio)+600.
    result=dict(state='single_fixed_generated_fullnative_Fishr_resource_gate',gate_passed=reserve<=43200.,rows=rows,sums=sums,Fishr_native_ratio=ratio,reserve_seconds=reserve,formula=plan['formula'],model_wall_and_CPU_cap_seconds=43200,RSS_KiB_cap=12582912,parameters=617612,batch=64,epochs=64,seed=20261029,active_steps_six_of_nine=True,synthetic_counter_fixture_is_not_1500_actual_updates=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,profile_wall=time.monotonic()-whole,profile_CPU=time.process_time()-cpu,maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_resource_rescue_grid=True)
    assert result['maxRSS_KiB']<=12582912
    write(PHASE/'RESOURCE_GATE.json',result);print(json.dumps(result))
if __name__=='__main__':main()
