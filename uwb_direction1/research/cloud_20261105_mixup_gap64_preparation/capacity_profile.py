"""Fixed synthetic capacity, including lazy slow state, validation and checkpoint."""
import time,resource,tempfile,json
from pathlib import Path
from common import B,ready,write
import engine as e
def main():
    c=ready();t=time.monotonic();c0=time.process_time();m,o,g=e.make();rng=e.np.random.default_rng(82)
    x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;rows=[]
    for i in range(c['profile_updates']):
        m.train();w=time.monotonic();p=time.process_time();row=e.step(m,o,x,y,1 if i<6 else 10)
        rows.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p,**row))
    m.eval();validation=[]
    with e.torch.inference_mode():
        for i in range(c['profile_validation_batches']):
            w=time.monotonic();p=time.process_time();z=m(e.torch.from_numpy(x));assert e.torch.isfinite(z).all()
            validation.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p))
    with tempfile.TemporaryDirectory() as d:
        w=time.monotonic();p=time.process_time();e.atomic_save(Path(d)/'state.pt',e.capture(m,o,g,1,8));checkpoint=dict(wall=time.monotonic()-w,cpu=time.process_time()-p)
    updates=64*((4700+63)//64);vb=64*((1000+63)//64);db=(1200+63)//64
    # Producer plus full fresh replay, two selected-output passes, first-update audit.
    projection={}
    for k in ['wall','cpu']:
        train=max(r[k] for r in rows);ev=max(r[k] for r in validation)
        projection[k]=c['profile_total_projection_safety']*((2*updates+2)*train+(2*vb+2*(db+16))*ev+130*checkpoint[k])
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    passed=projection['wall']<=c['real_wall_cap'] and projection['cpu']<=c['real_CPU_cap'] and rss<=c['profile_RSS_KiB']
    out=dict(state='synthetic_capacity_complete',gate_passed=passed,updates=rows,validation=validation,checkpoint=checkpoint,projected_full_producer_and_fresh64_replay_seconds=projection,safety_factor=c['profile_total_projection_safety'],maxRSS_KiB=rss,wall_seconds=time.monotonic()-t,CPU_seconds=time.process_time()-c0,full_parameters=617612,new_UWB_fit_started=False,no_UWB_arrays_or_labels_opened=True)
    write(B/'RESOURCE_RESULT.json',out);print(json.dumps(out))
if __name__=='__main__':main()
