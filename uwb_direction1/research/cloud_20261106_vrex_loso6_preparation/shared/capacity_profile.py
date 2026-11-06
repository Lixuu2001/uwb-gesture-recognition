"""One fixed generated-data VREx/native timing ratio; never opens UWB arrays."""
import time,copy,json,resource
from common import B,PHASE,ready,write
import engine as e

def main():
    ready();rng=e.np.random.default_rng(909);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12
    domains=e.np.repeat(e.c['train_users'],16);rows=[];whole=time.monotonic();cpu=time.process_time()
    # Fixed order and nine steps, including one explicit generated counter fixture at500.
    for arm in ['native','VREx']:
        m,o,g=e.make();m.train()
        for j in range(9):
            if arm=='VREx' and j==8:o.algorithm_updates=500
            w=time.monotonic();c=time.process_time()
            if arm=='native':
                o.zero_grad();z=m(e.torch.from_numpy(x));loss=e.native_bce(z,e.torch.as_tensor(y));loss.backward();e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step()
                fixture=False
            else:
                r=e.step(m,o,x,y,1,domains);fixture=(j==8)
                assert r['completed_updates_after']==(501 if fixture else j+1)
                if fixture:assert r['vrex_penalty_weight']==10. and r['optimizer_reset_this_update'] and o.count==1
            rows.append(dict(arm=arm,step=j,wall=time.monotonic()-w,cpu=time.process_time()-c,synthetic_counter_fixture500=fixture))
    sums={arm:{k:sum(r[k] for r in rows if r['arm']==arm) for k in ['wall','cpu']} for arm in ['native','VREx']}
    ratio=max(sums['VREx'][k]/sums['native'][k] for k in ['wall','cpu'])
    fixed_cost=30891.040083422995;reserve=1.25*fixed_cost*max(1.,ratio)+600.
    result=dict(state='fixed_synthetic_native_ratio_plus_actual_whole_MMD6_cost_gate',gate_passed=reserve<=43200.,rows=rows,sums=sums,VREx_native_ratio=ratio,reserve_seconds=reserve,formula='1.25*30891.040083422995*max(1,ratio)+600 <=43200',model_wall_and_CPU_cap_seconds=43200,RSS_KiB_cap=12582912,parameters=617612,batch=64,epochs=64,seed=20261029,synthetic_fixture_is_not_500_real_updates=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,profile_wall=time.monotonic()-whole,profile_CPU=time.process_time()-cpu,maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_resource_rescue_grid=True)
    assert result['maxRSS_KiB']<=12582912
    write(PHASE/'RESOURCE_GATE.json',result);print(json.dumps(result))
if __name__=='__main__':main()
