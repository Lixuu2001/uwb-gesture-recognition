"""Synthetic all-parameter Adam equation check; never loads any UWB values/labels.

Reference: Kingma/Ba Algorithm1, arxiv1412.6980, and PyTorch2.15 Adam docs.
This checks each update from its preceding FP32 state using an FP64 reference.
It is not an independent convolution/autograd implementation or accuracy test.
"""
from pathlib import Path
import json,hashlib,unittest
from native_adam100 import np,torch,make,optimizer,preprocess
U=2.0**-24
GAMMA=16*U/(1-16*U)
UNDERFLOW=16*float(np.finfo(np.float32).tiny)
LR=.001
B1=.9
B2=.999
EPS=1e-8
def reference(before,gradient,m0,v0,t):
 m=B1*m0+(1-B1)*gradient
 v=B2*v0+(1-B2)*gradient.square()
 # Covers coefficient conversion, lerp/multiply/add/square order differences,
 # including a conservative smallest-normal allowance for denormal flushing.
 em=GAMMA*(m0.abs()+gradient.abs())+UNDERFLOW
 ev=GAMMA*(v0.abs()+gradient.square())+UNDERFLOW
 scale=(1-B2**t)**.5
 low=((v-ev).clamp_min(0).sqrt()/scale+EPS)*(1-GAMMA)
 high=((v+ev).sqrt()/scale+EPS)*(1+GAMMA)
 denom=v.sqrt()/scale+EPS
 step=LR/(1-B1**t)
 expected=before-step*m/denom
 error=abs(step)*(em/low+m.abs()*(high-low)/low.square())
 error+=GAMMA*(before.abs()+abs(step)*(m.abs()+em)/low)+UNDERFLOW
 return m,v,em,ev,expected,error
class AdamEquations(unittest.TestCase):
 def test_all1265904_parameters_moments_bias_correction_and_BN_train(self):
  model=make();opt=optimizer(model);self.assertIs(type(opt),torch.optim.Adam)
  group=opt.param_groups[0]
  self.assertEqual((group['lr'],group['betas'],group['eps'],group['weight_decay']),(.001,(.9,.999),1e-8,0))
  for key in ['amsgrad','foreach','fused','maximize','capturable','differentiable']:
   self.assertIs(group[key],False,key)
  self.assertEqual(sum(p.numel() for p in model.parameters()),1265904)
  initial_BN=int(model.conv1[1].num_batches_tracked)
  rng=np.random.default_rng(20261029);rows=[]
  for t in [1,2]:
   raw=rng.normal(size=(20,3,90,189)).astype(np.float32)
   labels=torch.arange(20)%12;model.train();opt.zero_grad(set_to_none=True)
   loss=torch.nn.functional.cross_entropy(model(preprocess(raw)),labels)
   self.assertTrue(torch.isfinite(loss));loss.backward();expected={}
   for name,p in model.named_parameters():
    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    state=opt.state[p];before=p.detach().double().clone();g=p.grad.detach().double().clone()
    m0=state['exp_avg'].double().clone() if state else torch.zeros_like(before)
    v0=state['exp_avg_sq'].double().clone() if state else torch.zeros_like(before)
    expected[name]=reference(before,g,m0,v0,t)
   opt.step();count=0;max_ratio=0.;max_error=0.
   for name,p in model.named_parameters():
    state=opt.state[p];m,v,em,ev,ref,bound=expected[name]
    self.assertEqual(float(state['step']),float(t))
    self.assertTrue(torch.isfinite(p).all(),name)
    self.assertTrue(((state['exp_avg'].double()-m).abs()<=em).all(),name+' first moment')
    self.assertTrue(((state['exp_avg_sq'].double()-v).abs()<=ev).all(),name+' second moment')
    err=(p.detach().double()-ref).abs()
    self.assertTrue((err<=bound).all(),name+' parameter update')
    max_ratio=max(max_ratio,float((err/bound).max()));max_error=max(max_error,float(err.max()));count+=p.numel()
   self.assertEqual(count,1265904)
   rows.append(dict(update=t,parameter_elements_checked=count,loss=float(loss.detach()),maximum_parameter_error=max_error,maximum_error_over_predeclared_bound=max_ratio))
  self.assertEqual(int(model.conv1[1].num_batches_tracked),initial_BN+2)
  result=dict(state='pass_synthetic_full_B20_all_parameter_standard_Adam_equations',seed=20261029,updates=rows,batch_size=20,all_layers_trainable=True,BN_train=True,gamma_n=16,unit_roundoff=U,underflow_allowance=UNDERFLOW,tolerance_empirically_fitted=False,UWB_values_or_labels_read=False,actual_UWB_fit_started=False,author_radar_optimizer_code_parity_claim=False,independent_convolution_or_autograd=False)
  Path(__file__).with_name('ADAM_EQUATION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)
if __name__=='__main__':unittest.main(verbosity=2)
