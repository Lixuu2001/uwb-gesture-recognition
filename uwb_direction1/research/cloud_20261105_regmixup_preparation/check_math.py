"""Actual author AST CPU comparison, normalization, gradient and BN concatenation."""
import ast,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from prototype import batch,weighted_ce,native_ovr_bce_transfer
B=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.monotonic()
    env=dict(torch=torch,np=np)
    t=ast.parse((B/'author/mixup.py').read_text());fun=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='mixup_data'];assert len(fun)==1
    class CPU(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                node.func.attr='to';node.args=[ast.Constant('cpu')];node.keywords=[]
            return node
    code=ast.fix_missing_locations(CPU().visit(ast.Module(body=fun,type_ignores=[])));exec(compile(code,'author_mixup_CPU_device_only_adapter','exec'),env)
    t=ast.parse((B/'author/regmixup.py').read_text());f=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='regmixup_criterion'];exec(compile(ast.Module(body=f,type_ignores=[]),'author_regmixup_criterion','exec'),env)
    rng=np.random.default_rng(61);x=torch.from_numpy(rng.normal(size=(64,4,9)).astype('float32'));y=torch.arange(64)%12
    np.random.seed(20261029);torch.manual_seed(20261029);xx,a,b,lam,p=batch(x,y)
    np.random.seed(20261029);torch.manual_seed(20261029);mixed,ya,yb,author_lam=env['mixup_data'](x,y,20.,20.)
    assert lam==author_lam;torch.testing.assert_close(xx,torch.cat([x,mixed]),rtol=0,atol=0);torch.testing.assert_close(a,torch.cat([y,ya]),rtol=0,atol=0);torch.testing.assert_close(b,torch.cat([y,yb]),rtol=0,atol=0)
    logits=torch.tensor(rng.normal(size=(128,12)),dtype=torch.float64,requires_grad=True)
    ours=weighted_ce(logits,a,b,lam);ref=env['regmixup_criterion'](F.cross_entropy,logits,a,b,lam);torch.testing.assert_close(ours,ref,rtol=0,atol=0)
    target=lam*F.one_hot(a,12).double()+(1-lam)*F.one_hot(b,12).double();soft=-(target*F.log_softmax(logits,dim=-1)).sum(-1).mean()
    halfsum=.5*(F.cross_entropy(logits[:64],y)+lam*F.cross_entropy(logits[64:],y)+(1-lam)*F.cross_entropy(logits[64:],yb))
    torch.testing.assert_close(ours,soft,rtol=1e-14,atol=1e-14);torch.testing.assert_close(ours,halfsum,rtol=1e-14,atol=1e-14)
    actual=torch.autograd.grad(ours,logits,retain_graph=True)[0];expected=torch.autograd.grad(soft,logits)[0];torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    bn=torch.nn.BatchNorm1d(4);bn.train();bn(xx);assert int(bn.num_batches_tracked)==1 and len(xx)==128
    # Native OvR-BCE adaptation checked independently; not author's CE benchmark.
    ovr=native_ovr_bce_transfer(logits,a,b,lam)
    hard_a=F.one_hot(a,12).double();hard_b=F.one_hot(b,12).double()
    ovr_ref=lam*F.binary_cross_entropy_with_logits(logits,hard_a)+(1-lam)*F.binary_cross_entropy_with_logits(logits,hard_b)
    torch.testing.assert_close(ovr,ovr_ref,rtol=1e-14,atol=1e-14)
    go=torch.autograd.grad(ovr,logits,retain_graph=True)[0];gr=torch.autograd.grad(ovr_ref,logits)[0]
    torch.testing.assert_close(go,gr,rtol=1e-14,atol=1e-14)
    q=dict(state='pass_conditional_RegMixup_arithmetic_not_real_experiment',author_input_mix_and_cat_targets_exact=True,author_CE_value_and_gradient_exact=True,code_loss_mean_is_half_sum_of_equal_sized_real_and_mixed_batches=True,BN_single_concatenated_forward_one_update=True,class_default_and_CIFAR10_recipe_alpha_beta=[20,20],CIFAR100_recipe_differs=[10,10],sampling_fixed_points_follow_author_code=True,no_UWB_arrays_or_labels_opened=True,no_real_UWB_protocol_registered=True,no_new_background_training_started=True,current_Manifold_process_untouched=True,wall_seconds=time.monotonic()-started)
    q['proposed_native_OvR_BCE_value_and_gradient_checked']=True
    (B/'SYNTHETIC_MATH_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
