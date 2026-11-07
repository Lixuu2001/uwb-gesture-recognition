"""Independent NumPy/DomainBed arithmetic, full native gradients, RNG/BN and disk replay checks."""
from pathlib import Path
import ast,copy,json,time,tempfile,types
import native_engine as e
from andmask_core import combine,gradients_from_pooled_logits
P=Path(__file__).resolve().parent

def main():
    begin=time.monotonic();cpu=time.process_time();t=e.torch;n=e.np
    rng=n.random.default_rng(4421)
    arrays=[rng.normal(size=(4,5,7)).astype('float32'),rng.normal(size=(4,12)).astype('float32')]
    arrays[0][:,0,0]=0;arrays[0][:,0,1]=[1,-1,1,-1];arrays[1][:,0]=1
    groups=[[t.tensor(a[j]) for a in arrays] for j in range(4)]
    actual,diag=combine(groups,[13,17,19,15]);maximum=0.
    for a,got in zip(arrays,actual):
        mask=(n.abs(n.sign(a).mean(axis=0))>=1).astype('float32')
        ref=mask*n.tensordot(n.array([13,17,19,15],dtype='float64')/64,a,axes=1)/(1e-10+mask.mean())
        maximum=max(maximum,float(n.max(n.abs(got.numpy()-ref))));n.testing.assert_allclose(got.numpy(),ref,rtol=3e-6,atol=3e-7)
    tree=ast.parse((P/'author_reference/algorithms.py').read_text())
    cls=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name=='ANDMask')
    method=copy.deepcopy(next(z for z in cls.body if isinstance(z,ast.FunctionDef) and z.name=='mask_grads'))
    ns={'torch':t};exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned_MIT_DomainBed_mask_grads','exec'),ns)
    params=[t.nn.Parameter(t.zeros_like(a)) for a in groups[0]]
    ns['mask_grads'](types.SimpleNamespace(tau=1.),1.,list(zip(*groups)),params)
    ours,_=combine(groups,[1,1,1,1])
    for p,g in zip(params,ours):t.testing.assert_close(p.grad,g,rtol=2e-6,atol=2e-7)
    x=t.randn(64,90,567,generator=t.Generator().manual_seed(7821));y=t.arange(64)%12;domains=t.tensor([1]*13+[2]*17+[3]*19+[4]*15)
    m,opt=e.make();initial=e.capture(m,opt);m.train();logits=m(x);parameters=list(m.parameters())
    target=t.nn.functional.one_hot(y,12).float();loss=t.nn.functional.binary_cross_entropy_with_logits(logits,target)
    native=t.autograd.grad(loss,parameters,retain_graph=True)
    parts=[]
    for u in range(1,5):
        take=domains==u;domainloss=t.nn.functional.binary_cross_entropy_with_logits(logits[take],target[take]);parts.append(t.autograd.grad(domainloss,parameters,retain_graph=True))
    _,computed,diagnostic=gradients_from_pooled_logits(logits,y,domains,parameters)
    full_max=0.;reconstruct_max=0.
    for k,g in enumerate(computed):
        stack=t.stack([row[k] for row in parts]);mask=stack.sign().mean(0).abs().ge(1).float()
        mean=sum(row[k]*(count/64) for row,count in zip(parts,[13,17,19,15]))
        expected=mask*mean/(1e-10+mask.mean());t.testing.assert_close(g,expected,rtol=2e-5,atol=1e-6)
        full_max=max(full_max,float((g-expected).abs().max()));reconstruct_max=max(reconstruct_max,float((mean-native[k]).abs().max()))
        t.testing.assert_close(mean,native[k],rtol=1e-4,atol=2e-6)
    bn_after=e.semantic({k:v for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k});rng_after=t.get_rng_state().clone()
    baseline,bopt=e.make();e.restore(initial,baseline,bopt);baseline.train();baseline(x)
    assert bn_after==e.semantic({k:v for k,v in baseline.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k});assert t.equal(t.get_rng_state(),rng_after)
    m,opt=e.make();m.train()
    for _ in range(5):e.step(m,opt,x,y,domains)
    checkpoint=e.capture(m,opt)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'generated_fixture.pt';t.save(checkpoint,p);checkpoint=t.load(p,weights_only=False)
        for _ in range(2):e.step(m,opt,x,y,domains)
        expected=e.semantic(e.capture(m,opt));assert opt.count==7 and len(opt.slow)>0
        restored,ropt=e.make();e.restore(checkpoint,restored,ropt);restored.train()
        for _ in range(2):e.step(restored,ropt,x,y,domains)
        assert e.semantic(e.capture(restored,ropt))==expected
    # A missing source and singleton are current-batch facts, never zero-filled phantom gradients.
    small=t.randn(3,12,requires_grad=True);w=t.nn.Parameter(t.randn(12,12));z=small@w
    _,g,d=gradients_from_pooled_logits(z,t.tensor([0,1,2]),t.tensor([1,1,3]),[w]);assert d['present_domains']==[1,3] and d['domain_counts']==[2,1]
    zero,zd=combine([[t.zeros(3)],[t.zeros(3)]],[1,2]);assert t.equal(zero[0],t.zeros(3)) and zd['tensor_mask_densities']==[0.]
    result={'state':'pass_generated_fullnative_ANDmask_audits','parameters':617612,'shape':[64,90,567],'numpy_weighted_mask_max_difference':maximum,'all_native_parameter_masked_gradient_reference_max_difference':full_max,'sample_weighted_gradient_vs_native_globalBCE_max_difference':reconstruct_max,'pinned_DomainBed_uniform_counts_reference_passed':True,'one_native_BN_forward_and_dropout_RNG_preserved':True,'disk_resume_all_model_optimizer_slow_and_Python_NumPy_Torch_RNG_bitwise_equal':True,'lookahead_lazy_boundary6_checked':True,'absent_domain_singleton_and_zero_masks_checked':True,'real_UWB_samples_labels_or_IDs_read':False,'new_UWB_fit_started':False,'wall_seconds':time.monotonic()-begin,'CPU_seconds':time.process_time()-cpu}
    (P/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
