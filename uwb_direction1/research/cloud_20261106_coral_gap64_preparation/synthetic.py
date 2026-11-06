"""Exact pinned mean_cov, independent gradient and complete native CORAL transfer checks."""
from pathlib import Path
import ast,copy,json,time,tempfile
from common import B,ready,write
import engine as e
def reference_object():
 p=B/'author_reference/domainbed/algorithms.py';node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AbstractMMD')
 ns=dict(ERM=object,torch=e.torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),ns)
 obj=object.__new__(ns['AbstractMMD']);obj.kernel_type='mean_cov';return obj
def reference_penalty(features,domains):
 groups=[features[e.torch.as_tensor(domains)==u] for u in [3,4,5,6] if int((e.torch.as_tensor(domains)==u).sum())>=2];ref=reference_object()
 vals=[ref.mmd(x,y) for i,x in enumerate(groups) for y in groups[i+1:]]
 return sum(vals)/len(vals) if vals else features.sum()*0
def reference_step(model,opt,x,y,domains):
 box={}
 def save_features(module,args):box['features']=args[0]
 handle=model.head.register_forward_pre_hook(save_features)
 model.train();opt.zero_grad()
 try:z=model(e.torch.from_numpy(x))
 finally:handle.remove()
 ce=e.native_bce(z,e.torch.as_tensor(y,dtype=e.torch.long));pen=reference_penalty(box['features'],domains);loss=ce+pen
 loss.backward();e.torch.nn.utils.clip_grad_norm_(model.parameters(),2.);grads={k:p.grad.clone() for k,p in model.named_parameters()};opt.step()
 return dict(loss=float(loss.detach()),classification_loss=float(ce.detach()),penalty=float(pen.detach())),grads
def main():
 ready();start=time.monotonic();rng=e.np.random.default_rng(650);assert e.c['coral_gamma']==1.
 tree=ast.parse((B/'author_reference/domainbed/hparams_registry.py').read_text());defaults=[n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hparam' and len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='mmd_gamma' and isinstance(n.args[1],ast.Constant)];assert defaults==[1.]
 features=e.torch.from_numpy(rng.normal(size=(9,128))).requires_grad_();domains=e.np.array([3]*5+[4]*4)
 ours,groups,pairs=e.coral_penalty(features,domains);ref=reference_penalty(features,domains);assert groups==2 and pairs==1;e.torch.testing.assert_close(ours,ref,rtol=0,atol=0)
 grad=e.torch.autograd.grad(ours,features)[0];x,y=features[:5],features[5:];mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;delta=(xc.t()@xc)/4-(yc.t()@yc)/3;d=128
 gx=2*(mx-my)/(5*d)+4*(xc@delta)/(4*d*d);gy=-2*(mx-my)/(4*d)-4*(yc@delta)/(3*d*d)
 e.torch.testing.assert_close(grad,e.torch.cat([gx,gy]),rtol=1e-12,atol=1e-14)
 assert e.torch.autograd.gradcheck(lambda z:e.coral_penalty(z,domains)[0],(features,),eps=1e-6,atol=1e-7,rtol=1e-5,fast_mode=True)
 singleton=e.torch.from_numpy(rng.normal(size=(4,128))).requires_grad_();zero,ng,npair=e.coral_penalty(singleton,[3,4,5,6]);assert ng==0 and npair==0 and zero.item()==0;e.torch.testing.assert_close(e.torch.autograd.grad(zero,singleton)[0],e.torch.zeros_like(singleton),rtol=0,atol=0)
 partial=e.torch.from_numpy(rng.normal(size=(5,128))).requires_grad_();p,g,k=e.coral_penalty(partial,[3,3,4,4,5]);assert g==2 and k==1;e.torch.testing.assert_close(p,reference_object().mmd(partial[:2],partial[2:4]),rtol=0,atol=0)
 x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;domains=e.np.repeat([3,4,5,6],16)
 m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
 m.train();result=e.step(m,o,x,y,1,domains);after=copy.deepcopy(e.capture(m,o,g,0,1));grads={k:p.grad.clone() for k,p in m.named_parameters()}
 n,no,ng=e.make();e.restore(initial,n,no,ng);reference,rg=reference_step(n,no,x,y,domains)
 for key in ['loss','classification_loss','penalty']:assert result[key]==reference[key]
 assert e.semantic(after)==e.semantic(e.capture(n,no,ng,0,1))
 for key in grads:e.torch.testing.assert_close(grads[key],rg[key],rtol=0,atol=0)
 m,o,g=e.make();zero_initial=copy.deepcopy(e.capture(m,o,g,0,0));old=e.c['coral_gamma']
 try:e.c['coral_gamma']=0.;m.train();e.step(m,o,x,y,1,domains);zero_state=copy.deepcopy(e.capture(m,o,g,0,1))
 finally:e.c['coral_gamma']=old
 n,no,ng=e.make();e.restore(zero_initial,n,no,ng);n.train();no.zero_grad();logits=n(e.torch.from_numpy(x));e.native_bce(logits,e.torch.as_tensor(y)).backward();e.torch.nn.utils.clip_grad_norm_(n.parameters(),2.);no.step()
 assert e.semantic(zero_state)==e.semantic(e.capture(n,no,ng,0,1)) and e.semantic(zero_state['model'])!=e.semantic(after['model'])
 # Lazy Lookahead sixth update and actual disk recovery seventh update.
 m,o,g=e.make()
 for _ in range(5):m.train();e.step(m,o,x,y,1,domains)
 saved=copy.deepcopy(e.capture(m,o,g,1,5));digest=e.semantic(saved);m.train();left=e.step(m,o,x,y,1,domains);six=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow
 n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,y,1,domains);assert left==right and e.semantic(six)==e.semantic(e.capture(n,no,ng,1,6)) and e.semantic(saved)==digest
 with tempfile.TemporaryDirectory() as d:
     p=Path(d)/'native.pt';e.torch.save(six,p);disk=e.torch.load(p,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,y,1,domains);seven=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,y,1,domains);assert left==right and e.semantic(seven)==e.semantic(e.capture(n,no,ng,1,7))
 n.eval();before=copy.deepcopy(e.capture(n,no,ng,1,7));plain=e.native.make_model('gap',e.c['seed']);plain.load_state_dict(n.state_dict());plain.eval();e.restore(before,n,no,ng)
 with e.torch.inference_mode():e.torch.testing.assert_close(n(e.torch.from_numpy(x)),plain(e.torch.from_numpy(x)),rtol=0,atol=0)
 assert e.semantic(before)==e.semantic(e.capture(n,no,ng,1,7))
 n.train();assert e.step(n,no,x[:28],y[:28],1,domains[:28])['forward_batch']==28
 q=dict(state='pass',parameters=617612,coral_gamma=1.,feature_dim=128,seed=20261029,raw_forward_batch=64,pinned_default_gamma_AST=True,pinned_mean_cov_value_and_gradient_exact=True,independent_unbiased_covariance_and_mean_gradient_and_finite_differences_pass=True,valid_group_singleton_and_zero_pair_rules_pass=True,full_native_pinned_penalty_with_declared_pooled_BCE_adapter_all_BN_gradients_optimizer_RNG_exact=True,zero_gamma_complete_native_baseline_exact=True,penalty_changes_full_native_update=True,native_Lookahead_slow6_disk7_allRNG_recovery_exact=True,evaluation_clean_native_exact_no_RNG=True,tail28_passed=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,Focal_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
 write(B/'SYNTHETIC_RESULT.json',q);print(json.dumps(q))
if __name__=='__main__':main()
