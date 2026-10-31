"""Source-level synthetic compatibility, not a chosen/preregistered UWB experiment."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,copy,hashlib,json,random,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'vendor/oscnn'))
import numpy as np
import torch
from Classifiers.OS_CNN.OS_CNN import OS_CNN,build_layer_with_layer_parameter
from Classifiers.OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list,get_Prime_number_in_a_range
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_text())['files']:assert hashlib.sha256((B/'vendor/oscnn'/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
def make():
 torch.manual_seed(7);np.random.seed(7);random.seed(7)
 # Exact author multivariate example shape recipe, not constructor's univariate budget.
 layers=generate_layer_parameter_list(1,min(90//4,89),[8*128*567,5*128*256+2*256*128],in_channel=567)
 m=OS_CNN(layers,12,False);o=torch.optim.Adam(m.parameters(),lr=.001)
 return layers,m,o
def step(m,o,x,y):
 o.zero_grad();z=m(x);loss=torch.nn.functional.cross_entropy(z,y);loss.backward();o.step();return loss
def snap(m,o):return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),training=m.training,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
def restore(s,m,o):m.load_state_dict(s['model']);o.load_state_dict(s['optimizer']);m.train(s['training']);torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);check_source();start=time.monotonic()
 # Independent NumPy branch convolution verifies packed zero-mask SAME alignment.
 torch.manual_seed(13);spec=[(2,2,k) for k in [1,2,3,5]];layer=build_layer_with_layer_parameter(spec);x0=np.random.RandomState(13).randn(3,2,17).astype('float32');w=layer.conv1d.weight.detach().numpy();bias=layer.conv1d.bias.detach().numpy();expected=[];lo=0
 for _,nout,k in spec:
  mask=layer.weight_mask[lo:lo+nout].numpy();positions=np.flatnonzero(mask[0,0]);assert len(positions)==k
  kernel=w[lo:lo+nout,:,positions[0]:positions[-1]+1];pad=np.pad(x0,((0,0),(0,0),((k-1)//2,k//2)));ref=np.stack([np.einsum('bck,ock->bo',pad[:,:,i:i+k],kernel)+bias[lo:lo+nout] for i in range(17)],axis=-1);expected.append(ref);lo+=nout
 got=layer.conv1d(layer.padding(torch.from_numpy(x0))).detach().numpy();np.testing.assert_allclose(got,np.concatenate(expected,axis=1),rtol=2e-5,atol=2e-6)
 # Execute the original fit's unmodified inner body as an independent source path.
 tree=ast.parse((B/'vendor/oscnn/Classifiers/OS_CNN/OS_CNN_easy_use.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit');loop=next(n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='sample');body=compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(loop.body),type_ignores=[])),'pinned_original_OS_CNN_innerloop','exec')
 layers,m,o=make();g=torch.Generator().manual_seed(18);x=torch.randn(2,567,90,generator=g);y=torch.tensor([1,8]);losses=[]
 for _ in range(3):loss=step(m,o,x,y);assert torch.isfinite(loss);losses.append(float(loss.detach()))
 saved=snap(m,o);assert len(o.state)==14
 regrown=any(bool((l.conv1d.weight.detach()[l.weight_mask==0]!=0).any()) for l in m.layer_list);assert regrown
 layers2,m2,o2=make();ns=dict(torch_OS_CNN=m2,optimizer=o2,criterion=torch.nn.CrossEntropyLoss(),sample=(x,y));original=[]
 for _ in range(3):exec(body,ns);original.append(float(ns['output'].detach()))
 equal(losses,original);equal(saved,snap(m2,o2))
 for _ in range(2):step(m,o,x,y)
 after=snap(m,o);_,m3,o3=make();restore(saved,m3,o3);equal(saved,snap(m3,o3))
 for _ in range(2):step(m3,o3,x,y)
 equal(after,snap(m3,o3));m3.eval()
 with torch.inference_mode():z=m3(x)
 assert z.shape==(2,12) and torch.isfinite(z).all();assert all(not bool((l.conv1d.weight.detach()[l.weight_mask==0]!=0).any()) for l in m3.layer_list)
 primes=get_Prime_number_in_a_range(1,22);coverage=sorted({a+b+c-2 for a in primes for b in primes for c in [1,2]})
 result=dict(state='pass_source_probe_only',author_commit='f70626d3c4aad70a76ba7b4745f556df01044279',original_source_AST_inner_body_3updates_model_Adam_all_RNG_exact=True,restore_next_two_updates_model_Adam_all_RNG_exact=True,independent_NumPy_masked_branch_convolution_and_even_SAME=True,masked_weights_regrow_after_optimizer_and_zero_on_next_forward=True,author_model_unmodified=True,synthetic_batch_size=2,synthetic_seed=7,probe_only_not_real_training_configuration=True,input_shape=[567,90],multivariate_example_budget=[8*128*567,5*128*256+2*256*128],author_wrapper_kernel_upper_bound=22,actual_primes=primes,actual_maximum_RF=max(coverage),RF1to90_fully_covered=(set(range(1,91))<=set(coverage)),layer_output_channels=[sum(i[1] for i in layer) for layer in layers],total_parameters_including_fixed_masks=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),trainable_Adam_states=len(o.state),source_constructor_epochs=2000,source_multivariate_notebook_demo_epochs=500,no_real_epoch_budget_or_configuration_selected=True,cost_gate_not_run=True,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'SOURCE_PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
