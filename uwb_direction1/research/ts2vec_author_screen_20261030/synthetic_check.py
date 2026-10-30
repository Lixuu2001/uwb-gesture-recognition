"""Independent loss math and unmodified full-default author fit on fake data."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import ast,inspect,json,time
from types import SimpleNamespace
import numpy as np
import torch
from author_adapter import BASE,source_check,make_model,TS2Vec,hierarchical_contrastive_loss,instance_contrastive_loss,temporal_contrastive_loss
def contrast(a,b,kind):
 B,T,C=a.shape
 if (kind=='instance' and B==1) or (kind=='temporal' and T==1):return 0.
 values=np.concatenate([a,b],axis=0 if kind=='instance' else 1)
 if kind=='instance':values=values.transpose(1,0,2)
 n=values.shape[1];loss=[]
 for series in values:
  sim=series@series.T
  for i in range(n):
   positive=(i+n//2)%n;eligible=np.delete(sim[i],i);m=eligible.max();loss.append(-sim[i,positive]+m+np.log(np.exp(eligible-m).sum()))
 return float(np.mean(loss))
def hierarchy(a,b):
 values=[]
 while a.shape[1]>1:
  values.append(.5*contrast(a,b,'instance')+.5*contrast(a,b,'temporal'));B,T,C=a.shape
  a=a[:,:T//2*2].reshape(B,T//2,2,C).max(2);b=b[:,:T//2*2].reshape(B,T//2,2,C).max(2)
 if a.shape[1]==1:values.append(.5*contrast(a,b,'instance'))
 return float(np.mean(values))
def hierarchy_author_accumulation(a,b):
 # Author loss initializes torch.tensor(0.) (float32), then uses add_ even
 # with float64 inputs. Preserve each rounding boundary in this independent
 # reference; the original source and its optimization remain unchanged.
 value=np.float32(0.);d=0
 while a.shape[1]>1:
  value=np.float32(np.float64(value)+.5*contrast(a,b,'instance'))
  value=np.float32(np.float64(value)+.5*contrast(a,b,'temporal'));d+=1;B,T,C=a.shape
  a=a[:,:T//2*2].reshape(B,T//2,2,C).max(2);b=b[:,:T//2*2].reshape(B,T//2,2,C).max(2)
 if a.shape[1]==1:value=np.float32(np.float64(value)+.5*contrast(a,b,'instance'));d+=1
 return float(np.float32(np.float64(value)/d))
def main():
 assert not (BASE/'SYNTHETIC_CHECK.json').exists();started=time.monotonic();index=source_check()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 random=np.random.default_rng(408);cases=0
 for B,T in [(3,7),(2,8),(1,7),(3,1),(1,1)]:
  a=random.normal(size=(B,T,5));b=random.normal(size=(B,T,5));x=torch.tensor(a,requires_grad=True);y=torch.tensor(b,requires_grad=True)
  for fn,kind in [(instance_contrastive_loss,'instance'),(temporal_contrastive_loss,'temporal')]:np.testing.assert_allclose(float(fn(x,y).detach()),contrast(a,b,kind),rtol=1e-12,atol=1e-12)
  loss=hierarchical_contrastive_loss(x,y);assert loss.dtype==torch.float32
  np.testing.assert_allclose(float(loss.detach()),hierarchy_author_accumulation(a,b),rtol=0,atol=0)
  if B>1 and T>1:
   loss.backward();eps=1e-6;plus=a.copy();minus=a.copy();plus[0,1,2]+=eps;minus[0,1,2]-=eps
   np.testing.assert_allclose(float(x.grad[0,1,2]),(hierarchy(plus,b)-hierarchy(minus,b))/(2*eps),rtol=1e-5,atol=1e-7)
  cases+=1
 # Verify literal original automatic iteration expression without fitting600steps.
 tree=ast.parse((BASE/'vendor/ts2vec/ts2vec.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TS2Vec');fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
 expr=next(n.value for n in ast.walk(fit) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='n_iters' for t in n.targets) and isinstance(n.value,ast.IfExp))
 count=eval(compile(ast.Expression(expr),'author_default_count','eval'),{'train_data':SimpleNamespace(size=4700*90*567)})
 assert count==600
 model=make_model();parameters=sum(p.numel() for p in model._net.parameters());assert model.batch_size==16 and model.lr==.001 and model.temporal_unit==0 and model.max_train_length is None
 assert model._net.output_dims==320 and model._net.hidden_dims==64 and len(model._net.feature_extractor.net)==11 and int(model.net.n_averaged)==1
 initial={n:v.detach().clone() for n,v in model._net.state_dict().items()};snapshots=[];losses=[]
 def callback(m,loss):
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m._net.parameters());snapshots.append({n:v.detach().clone() for n,v in m._net.state_dict().items()});losses.append(loss)
 model.after_iter_callback=callback;data=np.random.default_rng(208).normal(size=(32,90,567)).astype(np.float32)
 # Unmodified original fit; two steps are a synthetic test, not an accuracy run.
 model.fit(data.copy(),n_iters=2)
 assert model.n_iters==2 and len(losses)==2 and int(model.net.n_averaged)==3
 changed=sum(not torch.equal(v,model._net.state_dict()[n]) for n,v in initial.items());assert changed>0
 for n,v in model.net.module.state_dict().items():torch.testing.assert_close(v,(initial[n]+snapshots[0][n]+snapshots[1][n])/3,rtol=1e-5,atol=1e-7)
 result=model.encode(data.copy(),encoding_window='full_series');assert result.shape==(32,320) and np.isfinite(result).all();model.net.eval()
 with torch.inference_mode():direct=model.net(torch.from_numpy(data.copy())).numpy().max(axis=1)
 np.testing.assert_allclose(result,direct,rtol=1e-5,atol=1e-6)
 report=dict(state='pass',synthetic_only=True,real_UWB_values_or_labels_read=False,real_training_started=False,author_commit=index['commit'],source_files_verified=len(index['files']),parameters=parameters,output_dims=320,hidden_dims=64,constructor_depth=10,actual_residual_blocks=11,default_batch_size=16,default_iterations_for_proposed_training_size=count,independent_numpy_loss_cases=cases,original_float32_hierarchy_accumulation_rounding_exact=True,finite_difference_loss_gradients_verified=True,unmodified_author_fit_synthetic_steps=2,all_parameter_gradients_finite=True,changed_state_tensors=changed,averaged_model_includes_initial_plus_two_updates=True,full_series_averaged_encoder_maxpool_verified=True,author_fit_checkpoint_resume_not_implemented=True,downstream_classifier_not_executed=True,wall_seconds=time.monotonic()-started,limitations='Full default architecture with2synthetic author updates, not600steps or a real dataset accuracy result. Original save/load preserves only averaged encoder; resumable raw/average/optimizer/RNG/sampler/IDs still needed in a future registered real driver. No exact full-training trajectory parity claim.')
 (BASE/'SYNTHETIC_CHECK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
