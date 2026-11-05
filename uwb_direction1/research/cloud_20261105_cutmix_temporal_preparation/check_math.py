"""Actual unmodified radar-author helper, exact layout/RNG and independent derivatives."""
from pathlib import Path
import ast,json,time
import torch
from torch.nn import functional as F
from prototype import batch,splice,native_bce,to_author_layout,from_author_layout
B=Path(__file__).resolve().parent
def author_functions():
    tree=ast.parse((B/'author/radar/train.py').read_text());wanted=['_apply_cutmix','_mixup_ce_loss'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(nodes)==2
    env=dict(torch=torch,F=F);exec(compile(ast.Module(body=nodes,type_ignores=[]),'unmodified_pinned_radar_author_helpers','exec'),env);return env
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);t=time.monotonic();ref=author_functions()
    count=dict(mixed=0,skipped=0)
    # Sentinel data checks the exact raw.permute(0,2,1,3).reshape layout, not merely shape.
    four=torch.arange(7*3*90*189,dtype=torch.float64).reshape(7,3,90,189);x=from_author_layout(four);y=torch.arange(7)%12
    torch.testing.assert_close(to_author_layout(x),four,rtol=0,atol=0)
    for radar in range(3):torch.testing.assert_close(x[:,:,radar*189:(radar+1)*189],four[:,radar,:,:],rtol=0,atol=0)
    for seed in range(8):
        before=x.clone();torch.manual_seed(seed);a,ya,yb,lam,meta=batch(x,y);rng_after=torch.get_rng_state().clone()
        torch.manual_seed(seed);draw=float(torch.rand(1).item())
        if draw<.8:b,ra,rb,rl=ref['_apply_cutmix'](to_author_layout(x),y,1.);b=from_author_layout(b);count['mixed']+=1
        else:b,ra,rb,rl=x,y,y,1.;count['skipped']+=1
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ya,ra,rtol=0,atol=0);torch.testing.assert_close(yb,rb,rtol=0,atol=0);assert lam==rl and meta['decision']==draw and torch.equal(rng_after,torch.get_rng_state());torch.testing.assert_close(x,before,rtol=0,atol=0)
    assert count['mixed'] and count['skipped']
    # Area/duration ratio is a label-mass rule; not proof of physical gesture semantics.
    index=torch.tensor([1,2,3,4,5,6,0]);z=torch.randn(7,9,6,dtype=torch.float64,requires_grad=True);w=torch.randn_like(z)
    for start,length in [(0,0),(0,9),(3,4),(8,1)]:
        mask=torch.zeros_like(z);mask[:,start:start+length,:]=1
        mixed=splice(z,index,start,length);expected=z*(1-mask)+z[index]*mask;torch.testing.assert_close(mixed,expected,rtol=0,atol=0)
        actual=torch.autograd.grad((mixed*w).sum(),z,retain_graph=True)[0];inverse=torch.argsort(index);closed=(1-mask)*w+mask*w[inverse];torch.testing.assert_close(actual,closed,rtol=0,atol=0)
    logits=torch.randn(7,12,dtype=torch.float64,requires_grad=True);a=y;b=y[index];lam=1-4/9
    ours=native_bce(logits,a,b,lam);ha=F.one_hot(a,12).double();hb=F.one_hot(b,12).double();hard=lam*F.binary_cross_entropy_with_logits(logits,ha)+(1-lam)*F.binary_cross_entropy_with_logits(logits,hb)
    torch.testing.assert_close(ours,hard,rtol=1e-14,atol=1e-14);torch.testing.assert_close(torch.autograd.grad(ours,logits,retain_graph=True)[0],torch.autograd.grad(hard,logits,retain_graph=True)[0],rtol=1e-14,atol=1e-14)
    ce=ref['_mixup_ce_loss'](logits,a,b,lam,.1);target=.9*(lam*ha+(1-lam)*hb)+.1/12;independent=-(target*F.log_softmax(logits,dim=-1)).sum(-1).mean();torch.testing.assert_close(ce,independent,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(ce,logits,retain_graph=True)[0],torch.autograd.grad(independent,logits)[0],rtol=1e-14,atol=1e-14)
    q=dict(state='pass_conditional_temporal_CutMix_arithmetic_not_UWB_experiment',actual_unmodified_radar_author_batch_targets_lambda_and_Torch_RNG_exact=True,native_three_radar_layout_exact=True,source_input_immutable=True,mixed_and_skip_branches=count,zero_full_partial_splice_and_cross_sample_Jacobian_exact=True,author_CE_label_smoothing_value_gradient_verified=True,proposed_native_OvR_BCE_value_gradient_verified=True,geometry='single_full_range_temporal_span_shared_across_all_three_radars_not_rectangular_2D_patch',alpha=1.,probability=.8,no_UWB_arrays_or_labels_opened=True,no_real_UWB_protocol_registered=True,no_new_background_training_started=True,current_RegMixup_configuration_untouched=True,wall_seconds=time.monotonic()-t)
    (B/'SYNTHETIC_MATH_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
