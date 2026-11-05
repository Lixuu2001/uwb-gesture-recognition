"""One fixed B64 native DomainBed_SD capacity measurement, synthetic input only."""
import json,resource,time,tempfile
from pathlib import Path
from common import B,ready
import engine as e
def main():
 c=ready();start=time.monotonic();cpu=time.process_time();m,o,g=e.make();rng=e.np.random.default_rng(82);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;updates=[]
 for i in range(8):
  m.train();w=time.monotonic();p=time.process_time();r=e.step(m,o,x,y,1);updates.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p,**r))
 assert o.count==8 and o.slow;m.eval();validation=[]
 with e.torch.inference_mode():
  for _ in range(3):
   w=time.monotonic();p=time.process_time();z=m(e.torch.from_numpy(x));assert e.torch.isfinite(z).all();validation.append(dict(wall=time.monotonic()-w,cpu=time.process_time()-p))
 with tempfile.TemporaryDirectory() as d:
  w=time.monotonic();p=time.process_time();e.torch.save(e.capture(m,o,g,1,8),Path(d)/'native.pt');checkpoint=dict(wall=time.monotonic()-w,cpu=time.process_time()-p)
 n=64*((4700+63)//64);v=64*((1000+63)//64);diagnostic=(1200+63)//64;projection={}
 for k in ['wall','cpu']:projection[k]=2*((2*n+2)*max(r[k] for r in updates)+(2*v+2*(diagnostic+16))*max(r[k] for r in validation)+130*checkpoint[k])
 measured_projection=dict(projection)
 for k in ['wall','cpu']:projection[k]=max(projection[k],c['prospective_native_capacity_floor'][k])
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;q=dict(state='conditional_full_native_DomainBed_SD_synthetic_capacity_complete',gate_passed=projection['wall']<=43200 and projection['cpu']<=43200 and rss<=12582912,updates=updates,validation=validation,checkpoint=checkpoint,projected_full64_producer_and_own_fresh64_replay_seconds=projection,measured_projection_seconds=measured_projection,prior_prototype_floor_applied=True,safety_factor=2.,maxRSS_KiB=rss,caps=dict(wall=43200,CPU=43200,RSS_KiB=12582912),parameters=617612,raw_forward_batch=64,sd_reg=.1,penalty="uncentered_mean_all_logits_squared",anneal_steps=0,seed=20261029,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False)
 (B/'RESOURCE_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
