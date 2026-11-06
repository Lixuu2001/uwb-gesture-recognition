"""Exact pinned Gaussian_MMD, independent gradient and complete native MMD transfer checks."""
from pathlib import Path
import ast,copy,json,time,tempfile
from common import B,PHASE,ready,write
import engine as e
def reference_object():
 p=PHASE/'shared/author_reference/domainbed/algorithms.py';node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AbstractMMD')
 ns=dict(ERM=object,torch=e.torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),ns)
 obj=object.__new__(ns['AbstractMMD']);obj.kernel_type='gaussian';return obj
def reference_penalty(features,domains):
 groups=[features[e.torch.as_tensor(domains)==u] for u in e.c['train_users'] if int((e.torch.as_tensor(domains)==u).sum())>=1];ref=reference_object()
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
 ready();start=time.monotonic();rng=e.np.random.default_rng(650);assert e.c['mmd_gamma']==1.
 tree=ast.parse((PHASE/'shared/author_reference/domainbed/hparams_registry.py').read_text());defaults=[n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hparam' and len(n.args)>=2 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='mmd_gamma' and isinstance(n.args[1],ast.Constant)];assert defaults==[1.]
 features=e.torch.from_numpy(rng.normal(scale=.005,size=(9,128))).requires_grad_();domains=e.np.array([e.c['train_users'][0]]*5+[e.c['train_users'][1]]*4)
 ours,groups,pairs=e.mmd_penalty(features,domains);ref=reference_penalty(features,domains);assert groups==2 and pairs==1;e.torch.testing.assert_close(ours,ref,rtol=0,atol=0)
 grad=e.torch.autograd.grad(ours,features)[0];x,y=features[:5],features[5:]
 def weighted_rbf(left,right):
  differences=left[:,None,:]-right[None,:,:];distance=differences.square().sum(-1);weighted=sum(g*e.torch.exp(-g*distance) for g in e.c['kernel_gammas']);return weighted,differences
 kxx,dxx=weighted_rbf(x,x);kyy,dyy=weighted_rbf(y,y);kxy,dxy=weighted_rbf(x,y);kyx,dyx=weighted_rbf(y,x)
 gx=-4*(kxx[:,:,None]*dxx).sum(1)/25+4*(kxy[:,:,None]*dxy).sum(1)/20
 gy=-4*(kyy[:,:,None]*dyy).sum(1)/16+4*(kyx[:,:,None]*dyx).sum(1)/20
 e.torch.testing.assert_close(grad,e.torch.cat([gx,gy]),rtol=1e-10,atol=1e-10)
 assert e.torch.autograd.gradcheck(lambda z:e.mmd_penalty(z,domains)[0],(features,),eps=1e-7,atol=1e-6,rtol=1e-4,fast_mode=True)
 node=next(n for n in ast.parse((PHASE/'shared/author_reference/domainbed/algorithms.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AbstractMMD');fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='gaussian_kernel');assert ast.literal_eval(fn.args.defaults[0])==e.c['kernel_gammas']
 singleton=e.torch.from_numpy(rng.normal(scale=.005,size=(4,128))).requires_grad_();v,ng,npair=e.mmd_penalty(singleton,e.c['train_users']);assert ng==4 and npair==6 and e.torch.isfinite(v);e.torch.testing.assert_close(v,reference_penalty(singleton,e.c['train_users']),rtol=0,atol=0)
 no_pairs=e.torch.from_numpy(rng.normal(size=(2,128))).requires_grad_();zero,ng,npair=e.mmd_penalty(no_pairs,[e.c['train_users'][0]]*2);assert ng==1 and npair==0 and zero.item()==0;e.torch.testing.assert_close(e.torch.autograd.grad(zero,no_pairs)[0],e.torch.zeros_like(no_pairs),rtol=0,atol=0)
 identical=e.pair_mmd(features,features);assert identical.item()==0
 assert e.c['kernel_gammas']==[.001,.01,.1,1.,10.,100.,1000.] and e.c['diagonal_terms_included'] and not e.c['final_penalty_clamped']
 x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;domains=e.np.repeat(e.c['train_users'],16)
 m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
 m.train();result=e.step(m,o,x,y,1,domains);after=copy.deepcopy(e.capture(m,o,g,0,1));grads={k:p.grad.clone() for k,p in m.named_parameters()}
 n,no,ng=e.make();e.restore(initial,n,no,ng);reference,rg=reference_step(n,no,x,y,domains)
 for key in ['loss','classification_loss','penalty']:assert result[key]==reference[key]
 assert e.semantic(after)==e.semantic(e.capture(n,no,ng,0,1))
 for key in grads:e.torch.testing.assert_close(grads[key],rg[key],rtol=0,atol=0)
 m,o,g=e.make();zero_initial=copy.deepcopy(e.capture(m,o,g,0,0));old=e.c['mmd_gamma']
 try:e.c['mmd_gamma']=0.;m.train();e.step(m,o,x,y,1,domains);zero_state=copy.deepcopy(e.capture(m,o,g,0,1))
 finally:e.c['mmd_gamma']=old
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
 q=dict(state='pass',parameters=617612,mmd_gamma=1.,feature_dim=128,seed=20261029,raw_forward_batch=64,pinned_default_gamma_AST=True,pinned_seven_Gaussian_MMD_value_and_gradient_exact=True,independent_biased_MMD_analytic_gradient_and_finite_differences_pass=True,valid_group_singleton_and_zero_pair_rules_pass=True,full_native_pinned_penalty_with_declared_pooled_BCE_adapter_all_BN_gradients_optimizer_RNG_exact=True,zero_gamma_complete_native_baseline_exact=True,penalty_changes_full_native_update=True,native_Lookahead_slow6_disk7_allRNG_recovery_exact=True,evaluation_clean_native_exact_no_RNG=True,tail28_passed=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,CORAL_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
 write(B/'SYNTHETIC_RESULT.json',q);print(json.dumps(q))
if __name__=='__main__':main()
