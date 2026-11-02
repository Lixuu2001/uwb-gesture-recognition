import copy, inspect, json
from unittest.mock import patch
import torch
from torch import nn
from unlabeled_adaptation import Adapter,entropy

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.gn=nn.GroupNorm(2,4);self.drop=nn.Dropout(.8);self.fc=nn.Linear(16,12)
    def forward(self,x):return self.fc(self.drop(self.gn(x).flatten(1)))

def equal(a,b):
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

def main():
    torch.set_num_threads(1);torch.manual_seed(14);m=Tiny().double();x=torch.randn(4,4,2,2,dtype=torch.float64);xx=x*1.3
    assert list(inspect.signature(Adapter.step).parameters)==['self','x']
    checks=[]
    for arm in ('frozen','entropy','filtered','sar'):
        a=Adapter(m,arm,margin=10,reset_threshold=-1);initial=copy.deepcopy(a.model.state_dict())
        expected=a.model(x).detach();pred,row=a.step(x);torch.testing.assert_close(pred,expected,rtol=0,atol=0)
        assert not a.model.drop.training
        for name,p in a.model.named_parameters():
            if not name.startswith('gn.') or arm=='frozen':torch.testing.assert_close(p,initial[name],rtol=0,atol=0)
        saved=a.state();p2,_=a.step(xx);after=a.state();b=Adapter(m,arm,margin=10,reset_threshold=-1);b.load(saved);p3,_=b.step(xx)
        torch.testing.assert_close(p2,p3,rtol=0,atol=0);equal(after['model'],b.state()['model'])
        a.reset();equal(initial,a.model.state_dict());assert not a.opt.state and a.ema is None
        # Prefix predictions are identical regardless of subsequently supplied data.
        b=Adapter(m,arm,margin=10,reset_threshold=-1);c=Adapter(m,arm,margin=10,reset_threshold=-1)
        for z in (x,xx):torch.testing.assert_close(b.step(z)[0],c.step(z)[0],rtol=0,atol=0)
        c.step(-x*9)
        checks.append(arm+'_prequential_whitelist_dropout_resume_reset_prefix')
    a=Adapter(m,'sar',margin=-1);s=a.state();_,r=a.step(x);equal(s['model'],a.model.state_dict());assert r['skip']=='empty_first'
    a=Adapter(m,'sar',margin=3,reset_threshold=-1);s=a.state();n=[0]
    def artificial(z):
        n[0]+=1;return entropy(z)+(10 if n[0]==2 else 0)
    with patch('unlabeled_adaptation.entropy',artificial):_,r=a.step(x)
    equal(s['model'],a.model.state_dict());assert r['skip']=='empty_second' and not a.opt.state
    a=Adapter(m,'sar',margin=10,reset_threshold=10);_,r=a.step(x);assert r['reset'] and a.ema is None and not a.opt.state;equal(a.initial,a.model.state_dict())
    a=Adapter(m,'entropy');s=a.state()
    try:a.step(x*float('nan'));raise AssertionError('Accepted nan')
    except ValueError:equal(s['model'],a.model.state_dict())
    a=Adapter(m,'entropy',margin=10);s=a.state();hook=a.params[0].register_hook(lambda g:g*float('nan'));_,r=a.step(x);hook.remove();equal(s['model'],a.model.state_dict());assert r['skip']=='nonfinite_first'
    # Independent autograd.grad implementation of one all-retained SAM step.
    ref=copy.deepcopy(m).eval();params=list(ref.gn.parameters());lr=.0005;rho=.05
    g=torch.autograd.grad(entropy(ref(x)).mean(),params);norm=torch.sqrt(sum((v*v).sum() for v in g));old=[p.detach().clone() for p in params]
    with torch.no_grad():
        for p,v in zip(params,g):p.add_(rho*v/(norm+1e-12))
    g2=torch.autograd.grad(entropy(ref(x)).mean(),params)
    with torch.no_grad():
        for p,v,o in zip(params,g2,old):p.copy_(o-lr*v)
    a=Adapter(m,'sar',margin=10,reset_threshold=-1);a.step(x)
    for p,v in zip(a.params,params):torch.testing.assert_close(p,v,atol=1e-14,rtol=1e-14)
    checks+=['empty_first','empty_second_rollback','collapse_reset','nonfinite_input','nonfinite_gradient','independent_float64_SAM']
    print(json.dumps(dict(passed=True,checks=checks)))
if __name__=='__main__':main()
