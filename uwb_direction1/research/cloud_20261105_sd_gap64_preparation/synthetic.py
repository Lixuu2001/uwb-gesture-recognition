"""Pinned DomainBed SD.update and declared native loss/optimizer transfer checks."""
from pathlib import Path
import ast,copy,json,time,tempfile,types
from common import B,ready
import engine as e
def reference_class(original_ce=False):
    path=B/'author_reference/domainbed/algorithms.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SD')
    adapter=e.torch.nn.functional if original_ce else types.SimpleNamespace(cross_entropy=e.native_bce)
    namespace=dict(ERM=object,torch=e.torch,F=adapter)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['SD']
class OptimizerAdapter:
    def __init__(self,opt,model,clip):self.opt=opt;self.model=model;self.clip=clip;self.grads=None
    def zero_grad(self):self.opt.zero_grad()
    def step(self):
        if self.clip:e.torch.nn.utils.clip_grad_norm_(self.model.parameters(),2.)
        self.grads={k:p.grad.clone() for k,p in self.model.named_parameters()}
        self.opt.step()
def reference_step(model,opt,x,y,original_ce=False,clip=True):
    cls=reference_class(original_ce);ref=object.__new__(cls);ref.sd_reg=.1;ref.predict=model
    proxy=OptimizerAdapter(opt,model,clip);ref.optimizer=proxy
    xx=e.torch.from_numpy(x);yy=e.torch.as_tensor(y,dtype=e.torch.long);mid=len(x)//2
    result=ref.update([(xx[:mid],yy[:mid]),(xx[mid:],yy[mid:])]);return result,proxy.grads
def main():
    ready();start=time.monotonic();rng=e.np.random.default_rng(650)
    # Unmodified SD.update under its original categorical CE on a small fixture.
    tiny=e.torch.nn.Linear(5,12);tx=rng.normal(size=(14,5)).astype('float32');ty=e.np.arange(14)%12
    twin=copy.deepcopy(tiny);a=e.torch.optim.SGD(tiny.parameters(),lr=.01);b=e.torch.optim.SGD(twin.parameters(),lr=.01)
    z=tiny(e.torch.from_numpy(tx));ce=e.torch.nn.functional.cross_entropy(z,e.torch.as_tensor(ty));pen=(z**2).mean();a.zero_grad();(ce+.1*pen).backward();a.step()
    result,grads=reference_step(twin,b,tx,ty,original_ce=True,clip=False)
    assert result==dict(loss=ce.item(),penalty=pen.item())
    for k,p in tiny.named_parameters():e.torch.testing.assert_close(p,dict(twin.named_parameters())[k],rtol=0,atol=0)
    # Full native forward, BN, all gradients, optimizer and all RNG.
    x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12
    m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
    m.train();ours=e.step(m,o,x,y,1);after=copy.deepcopy(e.capture(m,o,g,0,1));ours_grads={k:p.grad.clone() for k,p in m.named_parameters()}
    n,no,ng=e.make();e.restore(initial,n,no,ng);n.train();ref,ref_grads=reference_step(n,no,x,y)
    assert ref==dict(loss=ours['classification_loss'],penalty=ours['penalty'])
    assert e.semantic(after)==e.semantic(e.capture(n,no,ng,0,1))
    for k in ours_grads:e.torch.testing.assert_close(ours_grads[k],ref_grads[k],rtol=0,atol=0)
    # Independent penalty derivative/reduction and finite differences.
    logits=e.torch.from_numpy(rng.normal(size=(7,12))).requires_grad_();labels=e.torch.arange(7)%12
    _,penalty,_=e.loss_parts(logits,labels);grad=e.torch.autograd.grad(penalty,logits)[0]
    e.torch.testing.assert_close(grad,2*logits/84,rtol=1e-14,atol=1e-14)
    assert e.torch.autograd.gradcheck(lambda z:e.loss_parts(z,labels)[1],(logits,),eps=1e-6,atol=1e-7,rtol=1e-5)
    assert not e.torch.allclose(penalty,e.loss_parts(logits+3,labels)[1])
    # Zero coefficient is a same-native baseline control, no hidden style/RNG path.
    m,o,g=e.make();zero_initial=copy.deepcopy(e.capture(m,o,g,0,0));old=e.c['sd_reg']
    try:e.c['sd_reg']=0.;m.train();e.step(m,o,x,y,1);zero=copy.deepcopy(e.capture(m,o,g,0,1))
    finally:e.c['sd_reg']=old
    n,no,ng=e.make();e.restore(zero_initial,n,no,ng);n.train();no.zero_grad();z=n(e.torch.from_numpy(x));e.native_bce(z,e.torch.as_tensor(y)).backward();e.torch.nn.utils.clip_grad_norm_(n.parameters(),2.);no.step()
    assert e.semantic(zero)==e.semantic(e.capture(n,no,ng,0,1)) and e.semantic(zero['model'])!=e.semantic(after['model'])
    # Lazy Lookahead sixth update and actual disk recovery seventh update.
    m,o,g=e.make()
    for _ in range(5):m.train();e.step(m,o,x,y,1)
    saved=copy.deepcopy(e.capture(m,o,g,1,5));digest=e.semantic(saved);m.train();left=e.step(m,o,x,y,1);six=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow
    n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(six)==e.semantic(e.capture(n,no,ng,1,6)) and e.semantic(saved)==digest
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'native.pt';e.torch.save(six,p);disk=e.torch.load(p,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,y,1);seven=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(seven)==e.semantic(e.capture(n,no,ng,1,7))
    n.eval();before=copy.deepcopy(e.capture(n,no,ng,1,7));plain=e.native.make_model('gap',e.c['seed']);plain.load_state_dict(n.state_dict());plain.eval();e.restore(before,n,no,ng)
    with e.torch.inference_mode():e.torch.testing.assert_close(n(e.torch.from_numpy(x)),plain(e.torch.from_numpy(x)),rtol=0,atol=0)
    assert e.semantic(before)==e.semantic(e.capture(n,no,ng,1,7))
    n.train();assert e.step(n,no,x[:28],y[:28],1)['forward_batch']==28
    q=dict(state='pass',parameters=617612,sd_reg=.1,seed=20261029,raw_forward_batch=64,original_pinned_DomainBed_update_categorical_CE_fixture_exact=True,full_native_pinned_update_with_explicit_BCE_clip_optimizer_adapters_all_BN_gradients_optimizer_RNG_exact=True,independent_logit_penalty_gradient_and_finite_differences_pass=True,uncentered_mean_B_times12_reduction_verified=True,zero_coefficient_native_baseline_exact=True,penalty_changes_full_native_update=True,native_Lookahead_slow6_disk7_allRNG_recovery_exact=True,evaluation_clean_native_exact_no_RNG=True,tail28_passed=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,EFDMix_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
    (B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
