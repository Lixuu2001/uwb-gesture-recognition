"""Independent per-example autograd oracle and higher-order derivative checks."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from linear_gradient_variance import gradient_rows, variance_matching_penalty


def oracle(z,w,b,y,d):
    rows=[]
    for i in range(len(y)):
        loss=F.cross_entropy(F.linear(z[i:i+1],w,b),y[i:i+1])
        gw,gb=torch.autograd.grad(loss,(w,b),create_graph=True)
        rows.append(torch.cat((gw.flatten(),gb)))
    rows=torch.stack(rows)
    variances=torch.stack([rows[d==u].var(0,unbiased=False) for u in torch.unique(d)])
    return rows,variances.var(0,unbiased=False).mean()


def run():
    torch.set_num_threads(1);torch.manual_seed(20261004)
    z=torch.randn(9,4,dtype=torch.float64,requires_grad=True)
    w=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    b=torch.randn(3,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,2,1,2,0,2,0,1]);d=torch.tensor([3]*3+[4]*3+[5]*3)
    expected,reference=oracle(z,w,b,y,d);rows=gradient_rows(z,w,b,y)
    torch.testing.assert_close(rows,expected,rtol=1e-12,atol=1e-12)
    actual=variance_matching_penalty(z,w,b,y,d)
    torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
    a=torch.autograd.grad(actual,(z,w,b),retain_graph=True)
    e=torch.autograd.grad(reference,(z,w,b))
    for u,v in zip(a,e):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-12)
    assert torch.autograd.gradcheck(lambda zz,ww,bb:variance_matching_penalty(zz,ww,bb,y,d),(z,w,b),eps=1e-6,atol=1e-5,rtol=1e-4)
    perm=torch.randperm(len(y))
    torch.testing.assert_close(actual,variance_matching_penalty(z[perm],w,b,y[perm],d[perm]),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual,variance_matching_penalty(z.repeat(2,1),w,b,y.repeat(2),d.repeat(2)),rtol=1e-12,atol=1e-12)
    assert variance_matching_penalty(z,w,b,y,torch.zeros_like(d)).item()==0
    rejected=False
    try:variance_matching_penalty(z,w,b,y,torch.arange(len(y)))
    except ValueError:rejected=True
    assert rejected
    return {'state':'passed','scope':'synthetic float64 core only; no UWB results or full Fishr reproduction',
      'per_example_autograd_equal':True,'penalty_equal':True,'feature_weight_bias_higher_derivatives_equal':True,
      'finite_difference_gradcheck':True,'permutation_invariant':True,'whole_batch_duplication_invariant':True,
      'single_domain_zero':True,'singleton_domains_rejected':True,'torch':torch.__version__}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run();a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
