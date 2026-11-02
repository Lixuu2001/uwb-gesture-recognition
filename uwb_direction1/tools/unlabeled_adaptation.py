"""GN-only prequential entropy/SAR-style controls; explicit UWB variant.
References: Wang et al., Tent (ICLR 2021); Niu et al., SAR (ICLR 2023).
See research/SAR_LICENSE.txt and UNLABELED_ADAPTATION_REVIEW.md.
No labels, user IDs, or future batches enter Adapter.step.
"""
import copy, math
import torch


def entropy(logits):
    logp=logits.log_softmax(-1)
    return -(logp.exp()*logp).sum(-1)


class Adapter:
    def __init__(self, model, arm, lr=0.0005, momentum=0.9, rho=0.05,
                 margin=0.4*math.log(12), reset_threshold=0.2):
        if arm not in ('frozen','entropy','filtered','sar'):raise ValueError(arm)
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.arm=arm;self.rho=rho;self.margin=margin;self.reset_threshold=reset_threshold
        self.named=[]
        for name,module in self.model.named_modules():
            if isinstance(module,torch.nn.GroupNorm):
                for key in ('weight','bias'):
                    p=getattr(module,key);p.requires_grad_(arm!='frozen');self.named.append((name+'.'+key,p))
        if not self.named:raise ValueError('No GN affine parameters')
        self.params=[p for _,p in self.named]
        self.initial=copy.deepcopy(self.model.state_dict())
        self.opt=torch.optim.SGD(self.params,lr=lr,momentum=momentum)
        self.ema=None;self.steps=0;self.updates=0;self.resets=0

    def reset(self):
        self.model.load_state_dict(self.initial);self.opt.state.clear();self.opt.zero_grad(set_to_none=True);self.ema=None

    def state(self):
        return copy.deepcopy(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),ema=self.ema,
                         steps=self.steps,updates=self.updates,resets=self.resets))

    def load(self,s):
        self.model.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer'])
        for key in ('ema','steps','updates','resets'):setattr(self,key,s[key])

    def _backward(self,loss):
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):return False
        loss.backward()
        return all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.params)

    @torch.enable_grad()
    def step(self,x):
        if not torch.isfinite(x).all():raise ValueError('Nonfinite input rejected before mutation')
        self.model.eval();self.steps+=1
        logits=self.model(x);prediction=logits.detach().clone()
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite prediction')
        e=entropy(logits);mask=torch.ones_like(e,dtype=torch.bool) if self.arm=='entropy' else e<self.margin
        row=dict(batch=len(x),entropy=float(e.detach().mean()),first_count=int(mask.sum()),second_count=None,update=False,reset=False,skip=None)
        if self.arm=='frozen':return prediction,row
        if not mask.any():row['skip']='empty_first';return prediction,row
        if not self._backward(e[mask].mean()):row['skip']='nonfinite_first';self.opt.zero_grad(set_to_none=True);return prediction,row
        loss=e[mask].mean().detach()
        if self.arm=='sar':
            originals=[p.detach().clone() for p in self.params]
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in self.params]))
            if not torch.isfinite(norm):row['skip']='nonfinite_norm';self.opt.zero_grad(set_to_none=True);return prediction,row
            with torch.no_grad():
                for p in self.params:p.add_(p.grad,alpha=float(self.rho/(norm+1e-12)))
            try:
                second=entropy(self.model(x))[mask];keep=second<self.margin;row['second_count']=int(keep.sum())
                if not keep.any():row['skip']='empty_second';return prediction,row
                loss=second[keep].mean()
                if not self._backward(loss):row['skip']='nonfinite_second';return prediction,row
            finally:
                # Roll back every perturbation, even for empty/nonfinite second pass.
                with torch.no_grad():
                    for p,old in zip(self.params,originals):p.copy_(old)
                if row['skip'] is not None:self.opt.zero_grad(set_to_none=True)
        # Every effective gradient is finite before an optimizer step.
        before=self.state();self.opt.step()
        if not all(torch.isfinite(p).all() for p in self.params):
            self.load(before);self.opt.zero_grad(set_to_none=True);row['skip']='nonfinite_update';return prediction,row
        self.updates+=1;row['update']=True
        if self.arm=='sar':
            value=float(loss.detach());self.ema=value if self.ema is None else 0.9*self.ema+0.1*value
            if self.ema<self.reset_threshold:
                self.reset();self.resets+=1;row['reset']=True
        row['ema']=self.ema
        self.opt.zero_grad(set_to_none=True)
        return prediction,row
