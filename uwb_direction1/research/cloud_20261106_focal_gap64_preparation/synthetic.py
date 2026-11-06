"""Pinned torchvision/FAIR focal forward/backward and complete native state checks."""
from pathlib import Path
import ast,copy,json,time,tempfile
from common import B,ready
import engine as e
def reference_function(which='torchvision'):
 p=B/'author_reference'/which/'focal_loss.py'
 node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sigmoid_focal_loss')
 ns=dict(torch=e.torch,F=e.torch.nn.functional,_log_api_usage_once=lambda f:None)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),ns)
 return ns['sigmoid_focal_loss']
def reference_step(m,o,x,y):
 m.train();o.zero_grad();z=m(e.torch.from_numpy(x));t=e.torch.nn.functional.one_hot(e.torch.as_tensor(y,dtype=e.torch.long),12).to(z.dtype)
 value=reference_function()(z,t,alpha=.25,gamma=2,reduction='mean');value.backward();e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.)
 grads={n:p.grad.clone() for n,p in m.named_parameters()};o.step();return float(value.detach()),grads
def main():
 ready();start=time.monotonic();rng=e.np.random.default_rng(650)
 assert e.c['focal_alpha']==.25 and e.c['focal_gamma']==2. and e.c['focal_reduction']=='mean' and not e.c['focal_star']
 config=ast.parse((B/'author_reference/detectron/config.py').read_text());values={}
 for n in ast.walk(config):
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
   for dest in n.targets:
    if isinstance(dest,ast.Attribute) and dest.attr in ['LOSS_ALPHA','LOSS_GAMMA','PRIOR_PROB']:values[dest.attr]=n.value.value
 assert values==dict(LOSS_ALPHA=.25,LOSS_GAMMA=2.,PRIOR_PROB=.01)
 logits=e.torch.from_numpy(rng.normal(size=(7,12))).requires_grad_();y=e.torch.arange(7)%12;t=e.torch.nn.functional.one_hot(y,12).to(logits.dtype)
 ours=e.focal_loss(logits,y);pinned=reference_function()(logits,t,alpha=.25,gamma=2,reduction='mean')
 e.torch.testing.assert_close(ours,pinned,rtol=0,atol=0)
 grad=e.torch.autograd.grad(ours,logits)[0];sign=2*t-1;pt=e.torch.sigmoid(sign*logits);q=1-pt;ce=e.torch.nn.functional.softplus(-sign*logits);alpha=.25*t+.75*(1-t)
 independent=-sign*alpha*(2*pt*q**2*ce+q**3)/logits.numel()
 e.torch.testing.assert_close(grad,independent,rtol=1e-12,atol=1e-14)
 assert e.torch.autograd.gradcheck(lambda z:e.focal_loss(z,y),(logits,),eps=1e-6,atol=1e-7,rtol=1e-5)
 f32=logits.detach().float();e.torch.testing.assert_close(reference_function()(f32,t.float(),alpha=.25,gamma=2,reduction='none'),reference_function('fvcore')(f32,t.float(),alpha=.25,gamma=2,reduction='none'),rtol=0,atol=0)
 # gamma0/alpha-1 is only an engineering control; no UWB configuration sweep.
 e.torch.testing.assert_close(e.focal_loss(f32,y,alpha=-1,gamma=0),e.native_bce(f32,y),rtol=0,atol=0)
 x=rng.normal(size=(64,90,567)).astype('float32');labels=e.np.arange(64)%12
 m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
 m.train();result=e.step(m,o,x,labels,1);after=copy.deepcopy(e.capture(m,o,g,0,1));grads={k:p.grad.clone() for k,p in m.named_parameters()}
 n,no,ng=e.make();e.restore(initial,n,no,ng);ref,rg=reference_step(n,no,x,labels);assert ref==result['loss'] and e.semantic(after)==e.semantic(e.capture(n,no,ng,0,1))
 for key in grads:e.torch.testing.assert_close(grads[key],rg[key],rtol=0,atol=0)
 m,o,g=e.make()
 for _ in range(5):m.train();e.step(m,o,x,labels,1)
 saved=copy.deepcopy(e.capture(m,o,g,1,5));digest=e.semantic(saved);m.train();left=e.step(m,o,x,labels,1);six=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow
 n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,labels,1);assert left==right and e.semantic(six)==e.semantic(e.capture(n,no,ng,1,6)) and digest==e.semantic(saved)
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'native.pt';e.torch.save(six,p);disk=e.torch.load(p,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,labels,1);seven=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,labels,1);assert left==right and e.semantic(seven)==e.semantic(e.capture(n,no,ng,1,7))
 n.eval();before=copy.deepcopy(e.capture(n,no,ng,1,7));plain=e.native.make_model('gap',e.c['seed']);plain.load_state_dict(n.state_dict());plain.eval();e.restore(before,n,no,ng)
 with e.torch.inference_mode():e.torch.testing.assert_close(n(e.torch.from_numpy(x)),plain(e.torch.from_numpy(x)),rtol=0,atol=0)
 assert e.semantic(before)==e.semantic(e.capture(n,no,ng,1,7));n.train();assert e.step(n,no,x[:28],labels[:28],1)['forward_batch']==28
 from common import write
 q=dict(state='pass',parameters=617612,alpha=.25,gamma=2.,reduction='mean_B_times12',seed=20261029,raw_forward_batch=64,pinned_Detectron_default_AST_verified=True,pinned_torchvision_functional_full_native_all_BN_gradients_optimizer_all_RNG_exact=True,pinned_fvcore_f32_functional_exact=True,independent_scalar_derivative_and_finite_difference_pass=True,gamma0_alpha_minus1_BCE_control=True,native_Lookahead_slow6_disk7_recovery_exact=True,evaluation_clean_native_exact_no_RNG=True,tail28_passed=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
 write(B/'SYNTHETIC_RESULT.json',q);print(json.dumps(q))
if __name__=='__main__':main()
