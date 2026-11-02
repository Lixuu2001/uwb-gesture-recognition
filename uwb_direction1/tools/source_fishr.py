"""Source-calibrated last-layer variance intervention, not an original Fishr claim.

Uses conventional bias-corrected EMA, unlike the author's fixed1/(1-beta)
normalizer. All arms maintain identical types of state and source-only scale.
"""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from linear_gradient_variance import gradient_rows


def balanced_order(ids,users,c,held,epoch):
    groups=np.unique(users[ids]);assert len(groups)==4 and c['batch']==4*c['per_domain']
    pools=[ids[users[ids]==d] for d in groups];steps=math.ceil(max(map(len,pools))/c['per_domain']);orders=[]
    for d,pool in zip(groups,pools):
        seq=[];cycle=0
        while sum(map(len,seq))<steps*c['per_domain']:
            seq.append(np.random.default_rng(np.random.SeedSequence([c['seed'],held,epoch,int(d),cycle])).permutation(pool));cycle+=1
        orders.append(np.concatenate(seq)[:steps*c['per_domain']].reshape(steps,c['per_domain']))
    return np.stack(orders,axis=1).reshape(-1)


def permuted_domains(labels,domains,seed,held,step):
    y=labels.detach().cpu().numpy();d=domains.detach().cpu().numpy();p=d.copy()
    rng=np.random.default_rng(np.random.SeedSequence([seed,held,step,917]))
    for k in np.unique(y):
        ix=np.flatnonzero(y==k);p[ix]=rng.permutation(d[ix])
    return torch.from_numpy(p).to(domains.device)


class VarianceEMA:
    def __init__(self,beta):
        if not 0<=beta<1:raise ValueError('EMA beta out of range')
        self.beta=beta;self.steps=0;self.values={};self.scale=None;self.calibration=None
    def update(self,real,permuted):
        self.steps+=1;out={}
        for key,v in [('real',real),('permuted',permuted)]:
            raw=self.beta*self.values.get(key,torch.zeros_like(v))+(1-self.beta)*v
            self.values[key]=raw.detach().clone()
            smooth=raw/(1-self.beta**self.steps)
            out[key]=(smooth-smooth.mean(0)).square().mean()
        return out
    def state_dict(self):return copy.deepcopy(dict(beta=self.beta,steps=self.steps,values=self.values,scale=self.scale,calibration=self.calibration))
    def load_state_dict(self,s):
        if s['beta']!=self.beta:raise ValueError('EMA beta identity mismatch')
        for k,v in s.items():setattr(self,k,copy.deepcopy(v))


def source_step(net,opt,x,target,domains,ema,c,held,step,epoch,arm):
    encoded=net.encoder(x.reshape(len(x)*3,4,90,63)).reshape(len(x),3,-1)
    z=net.head[:4](encoded);logits=net.head[4](z);ce=F.cross_entropy(logits,target)
    rows=gradient_rows(z,net.head[4].weight,net.head[4].bias,target)
    shuffled=permuted_domains(target,domains,c['seed'],held,step)
    def variances(d):
        vs=[]
        for domain in torch.unique(d,sorted=True):
            g=rows[d==domain]
            if len(g)<2:raise ValueError('Insufficient domain examples')
            vs.append((g-g.mean(0)).square().mean(0))
        return torch.stack(vs)
    penalties=ema.update(variances(domains),variances(shuffled))
    active=epoch>c['warmup_epochs']
    if active and ema.scale is None:
        params=tuple(net.parameters())
        def norm(loss):
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=False)
            return torch.sqrt(sum(g.double().square().sum() for g in gs)).item()
        nc=norm(ce);nv=norm(penalties['real'])
        if not np.isfinite(nc+nv) or nv<=0 or nc<=0:raise ValueError('Invalid source calibration gradients')
        raw=c['calibration_norm_ratio']*nc/nv;ema.scale=float(np.clip(raw,*c['scale_bounds']))
        ema.calibration=dict(step=step,epoch=epoch,ce_gradient_norm=nc,real_penalty_gradient_norm=nv,unclipped_scale=raw,scale=ema.scale,clipped=raw!=ema.scale)
    loss=ce
    if active and arm!='balanced_erm':loss=loss+ema.scale*penalties['real' if arm=='real_variance' else 'permuted']
    if not torch.isfinite(loss):raise ValueError('Nonfinite source objective')
    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return {'ce':float(ce.detach()),'real_penalty':float(penalties['real'].detach()),'permuted_penalty':float(penalties['permuted'].detach()),'objective':float(loss.detach()),'scale':ema.scale}
