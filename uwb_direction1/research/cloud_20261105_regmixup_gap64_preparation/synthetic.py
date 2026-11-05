"""Full-width native initialization, author batch/forward, native loss and recovery checks."""
from pathlib import Path
import ast,copy,json,tempfile,time
import engine as e
from common import ready
from regmixup_math import batch,native_ovr_bce_transfer
B=Path(__file__).resolve().parent
def main():
    ready();started=time.monotonic();rng=e.np.random.default_rng(650)
    x=e.torch.from_numpy(rng.normal(size=(64,90,567)).astype('float32'));y=e.torch.arange(64)%12
    m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
    # Compare full native forward to actual released batch/target construction.
    env=dict(np=e.np,torch=e.torch)
    tree=ast.parse((B/'author/mixup.py').read_text());f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mixup_data'];assert len(f)==1
    class CPU(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':node.func.attr='to';node.args=[ast.Constant('cpu')];node.keywords=[]
            return node
    exec(compile(ast.fix_missing_locations(CPU().visit(ast.Module(body=f,type_ignores=[]))),'pinned_author_device_only','exec'),env)
    m.train();xx,a,b,lam,p=batch(x,y,20.,20.);before={n:int(z.num_batches_tracked) for n,z in m.named_modules() if isinstance(z,e.torch.nn.modules.batchnorm._BatchNorm)}
    z=m(xx);loss=native_ovr_bce_transfer(z,a,b,lam);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()};state_after=copy.deepcopy(m.state_dict());rng_after=e.torch.get_rng_state().clone()
    assert before and all(int(dict(m.named_modules())[n].num_batches_tracked)==v+1 for n,v in before.items())
    n,no,ng=e.make();e.restore(initial,n,no,ng);n.train();mx,ya,yb,al=env['mixup_data'](x,y,20.,20.);assert lam==al
    ref_logits=n(e.torch.cat([x,mx]));e.torch.testing.assert_close(z,ref_logits,rtol=0,atol=0)
    assert e.semantic(state_after)==e.semantic(n.state_dict()) and e.torch.equal(rng_after,e.torch.get_rng_state())
    target_a=e.torch.cat([y,ya]);target_b=e.torch.cat([y,yb])
    fa=e.torch.nn.functional.one_hot(target_a,12).float();fb=e.torch.nn.functional.one_hot(target_b,12).float()
    independent=lam*e.torch.nn.functional.binary_cross_entropy_with_logits(ref_logits,fa)+(1-lam)*e.torch.nn.functional.binary_cross_entropy_with_logits(ref_logits,fb)
    e.torch.testing.assert_close(loss,independent,rtol=1e-6,atol=1e-7);independent.backward()
    maximum_grad_difference=0.
    for name,p in n.named_parameters():
        e.torch.testing.assert_close(grads[name],p.grad,rtol=2e-5,atol=2e-7)
        maximum_grad_difference=max(maximum_grad_difference,float((grads[name]-p.grad).abs().max()))
    # Complete optimizer slow-state boundary plus exact RNG replay; no UWB samples.
    m,o,g=e.make()
    for _ in range(5):m.train();e.step(m,o,x.numpy(),y.numpy(),1)
    saved=copy.deepcopy(e.capture(m,o,g,5,5));saved_digest=e.semantic(saved);m.train();left=e.step(m,o,x.numpy(),y.numpy(),1);after=copy.deepcopy(e.capture(m,o,g,6,6))
    assert o.count==6 and o.slow and e.semantic(saved)==saved_digest
    n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x.numpy(),y.numpy(),1)
    assert left==right and e.semantic(after)==e.semantic(e.capture(n,no,ng,6,6)) and e.semantic(saved)==saved_digest
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'synthetic_native.pt';e.torch.save(after,path);loaded=e.torch.load(path,map_location='cpu',weights_only=True)
        m.train();left=e.step(m,o,x.numpy(),y.numpy(),1);finish=copy.deepcopy(e.capture(m,o,g,7,7));n,no,ng=e.make();e.restore(loaded,n,no,ng);n.train();right=e.step(n,no,x.numpy(),y.numpy(),1)
        assert left==right and e.semantic(finish)==e.semantic(e.capture(n,no,ng,7,7))
    q=dict(state='pass',parameters=617612,raw_batch=64,concatenated_forward_batch=128,alpha_beta=[20,20],seed=20261029,matched_native_noMixup_initial_model_exact=True,pinned_author_device_only_batch_native_logits_and_BN_buffers_exact=True,each_native_BN_one_update=True,native_OvRBCE_gradient_agrees_with_independent_weighted_hard_target_loss=True,max_gradient_absolute_difference=maximum_grad_difference,full_optimizer_slow_buffer_and_Python_NumPy_Torch_order_RNG_disk_resume_exact=True,saved_optimizer_fixture_immutable=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,current_Manifold_configuration_untouched=True,wall_seconds=time.monotonic()-started)
    (B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
