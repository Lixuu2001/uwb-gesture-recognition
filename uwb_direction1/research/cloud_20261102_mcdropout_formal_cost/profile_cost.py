import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,resource,signal,copy
from guard import ready,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 p=ready();out=B/'profile01';out.mkdir(exist_ok=False);(out/'POLICY.json').write_bytes((B/'POLICY.json').read_bytes());(out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),profiler_sha256=sha(__file__),cost_hashes_sha256=sha(B/'COST_HASHES.json'),policy_fixed_before_timing=True,real_UWB_values_labels_or_fit=False,trained_UWB_checkpoint_loaded=False),indent=2)+'\n');resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 sys.path.insert(0,'/workspace/uwb-mcdropout-source-probe-20261102');from mc_dropout import torch,np,dropout_only
 sys.path.insert(0,'/workspace/uwb-timemil-20261029');from adapter import make_model
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();m=make_model('timemil',p['seed']);assert sum(v.numel() for v in m.parameters())==850973
 rng=np.random.default_rng(p['seed']);x=torch.from_numpy((.1*rng.normal(size=(64,90,567))).astype('float32'));before=copy.deepcopy(m.state_dict());forward=[];outputs=[];states=[]
 with dropout_only(m) as nodes,torch.no_grad():
  assert len(nodes)==3 and all(n['p']==.2 for n in nodes);torch.manual_seed(p['seed'])
  for _ in range(8):
   t=time.monotonic();z=m(x);link=torch.sigmoid(z).numpy();forward.append(time.monotonic()-t);assert np.isfinite(link).all()
 raw=rng.normal(size=(100,2200,12)).astype('float32');mean=torch.sigmoid(torch.from_numpy(raw)).numpy().mean(0,dtype='float64');rs=torch.stack([torch.get_rng_state() for _ in range(100)])
 for i in range(3):
  t=time.monotonic();np.savez_compressed(out/f'outputs{i}.npz',logits=raw,mean_sigmoid=mean);outputs.append(time.monotonic()-t)
  t=time.monotonic();torch.save(dict(model=m.state_dict(),rng_states=rs,modes={n:v.training for n,v in m.named_modules()}),out/f'states{i}.pt');states.append(time.monotonic()-t)
 for n,v in m.state_dict().items():assert torch.equal(v,before[n])
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r=select(p,forward,outputs,states,rss);r.update(policy=p,full_random_TimeMIL850973=True,learned_rate_nodes=nodes,model_BN_buffers_unchanged=True,forward_seconds=forward,output_save_seconds=outputs,state_save_seconds=states,max_RSS_KiB=rss,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,real_UWB_values_labels_or_fit=False,trained_UWB_checkpoint_loaded=False,real_pilot_started=False)
 (out/'COST_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
