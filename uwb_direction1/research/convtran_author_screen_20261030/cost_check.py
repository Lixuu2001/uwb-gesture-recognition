"""Fixed resource-only synthetic test of64width and literal CLI16 defaults."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import hashlib,json,math,resource,time
import torch
from author_core import BASE,source_check,make_model as make_full,optimizer
from cli_core import make_model as make_cli
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 assert not (BASE/'COST_CHECK.json').exists();policy=json.loads((BASE/'COST_POLICY.json').read_text())
 for path,h in json.loads((BASE/'COST_HASHES.json').read_text()).items(): assert sha(BASE/path)==h,path
 source_check();assert json.loads((BASE/'SYNTHETIC_CHECK.json').read_text())['state']=='pass'
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();data=torch.randn(16,567,90,generator=torch.Generator().manual_seed(2048));labels=torch.arange(16)%12;profiles=[]
 for name,constructor in [('paper_width64_author_core',make_full),('literal_author_CLI_defaults16',make_cli)]:
  model,ns,cfg=constructor();opt=optimizer(model,ns);criterion=ns['NoFussCrossEntropyLoss'](reduction='none');parameters=sum(p.numel() for p in model.parameters());times={}
  for phase in ('train','validation'):
   samples=[];model.train(phase=='train')
   for step in range(4):
    start=time.monotonic()
    if phase=='train':
     opt.zero_grad(set_to_none=True);loss=criterion(model(data),labels).mean();assert torch.isfinite(loss);loss.backward()
     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),4.);opt.step()
    else:
     with torch.inference_mode(): assert torch.isfinite(model(data)).all()
    seconds=time.monotonic()-start
    if step: samples.append(seconds)
   times[phase]=samples
  estimate=2*(100*(math.ceil(4700/16)*max(times['train'])+math.ceil(1000/16)*max(times['validation']))+math.ceil(1200/16)*max(times['validation']))+120
  profile=dict(name=name,config=cfg,parameters=parameters,fullbatch_times=times,estimated_seconds=estimate,wall_and_CPU_gate_passed=estimate<=43200,optimizer_steps=4)
  profiles.append(profile);print(json.dumps(profile),flush=True);del model,opt,ns,criterion
 selected=next((p['name'] for p in profiles if p['wall_and_CPU_gate_passed']),None)
 result=dict(state='synthetic_cost_screen_complete',policy_sha256=sha(BASE/'COST_POLICY.json'),profiles=profiles,selected_by_resource_only=selected,real_data_or_labels_read=False,real_training_started=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitation='Synthetic one-thread timing while original TSLANet also uses one of five CPUs; estimator uses2xmaxsteadyfullbatch plus120seconds. Passing is a prospective resource estimate, not a guarantee or accuracy result.')
 (BASE/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(state=result['state'],selected=selected)))
if __name__=='__main__': main()
