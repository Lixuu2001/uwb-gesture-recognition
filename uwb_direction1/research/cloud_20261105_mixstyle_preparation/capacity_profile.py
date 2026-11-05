"""One pre-fixed B64 full-native synthetic capacity probe; no UWB arrays/labels."""
from pathlib import Path
import json,time,resource,tempfile
import engine as e
from common import ready
B=Path(__file__).resolve().parent
def main():
    ready();t=time.monotonic();c0=time.process_time();m,o,g=e.make();rng=e.np.random.default_rng(82)
    x=e.torch.from_numpy(rng.normal(size=(64,90,567)).astype('float32'));y=e.torch.arange(64)%12;rows=[]
    for i in range(8):
        m.train();w=time.monotonic();p=time.process_time();row=e.step(m,o,x.numpy(),y.numpy(),1);rows.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p,**row))
    assert o.count==8 and o.slow
    m.eval();validation=[]
    with e.torch.inference_mode():
        for _ in range(3):
            w=time.monotonic();p=time.process_time();z=m(x);assert e.torch.isfinite(z).all();validation.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p))
    with tempfile.TemporaryDirectory() as d:
        w=time.monotonic();p=time.process_time();e.torch.save(e.capture(m,o,g,1,8),Path(d)/'native.pt');checkpoint=dict(wall=time.monotonic()-w,cpu=time.process_time()-p)
    updates=64*((4700+63)//64);vb=64*((1000+63)//64);db=(1200+63)//64;projection={}
    for k in ['wall','cpu']:
        projection[k]=2.*((2*updates+2)*max(r[k] for r in rows)+(2*vb+2*(db+16))*max(r[k] for r in validation)+130*checkpoint[k])
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;gate=all(v<=43200 for v in projection.values()) and rss<=12582912
    q=dict(state='conditional_full_native_MixStyle_synthetic_capacity_complete',gate_passed=gate,updates=rows,validation=validation,checkpoint=checkpoint,projected_full64_producer_and_own_fresh64_replay_seconds=projection,safety_factor=2.,caps=dict(wall=43200,CPU=43200,RSS_KiB=12582912),maxRSS_KiB=rss,wall_seconds=time.monotonic()-t,CPU_seconds=time.process_time()-c0,parameters=617612,raw_batch=64,effective_forward_batch=64,alpha=.1,probability=.5,seed=20261029,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,preregistration_for_real_fit_complete=False)
    (B/'RESOURCE_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
